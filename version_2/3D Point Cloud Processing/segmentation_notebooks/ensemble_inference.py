import sys, os, io
import numpy as np
import laspy
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

TARGET_CLASS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREP = {
    "voxel_size": 0.01, "normal_radius": 0.05, "normal_max_nn": 30,
    "sor_enabled": True, "sor_k": 16, "sor_std": 2.0,
    "cluster_filter_enabled": True, "cluster_eps": 0.03, "cluster_min_size": 100,
    "geom_radius": 0.06,
}

ENSEMBLE_CHECKPOINTS = [
    ("ptv2",   "../checkpoints/PointCloudDataset_SlidingWindow_9_channel/PointTransformerV2_best.pth"),
    ("kpconv", "../checkpoints/KPConv_12_channel/KPConv_best.pth"),
]

LAS_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/test/sample_data_0046.las"


from torch_cluster import knn as tc_knn
from torch_scatter import scatter_softmax, scatter_add, scatter_max, scatter_mean


class GVA(nn.Module):
    def __init__(self, ch, g=6, k=16):
        super().__init__()
        assert ch % g == 0
        self.k, self.g, self.gc = k, g, ch // g
        self.q  = nn.Linear(ch, ch); self.kk = nn.Linear(ch, ch); self.v = nn.Linear(ch, ch)
        self.pm = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.pb = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.w  = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, g))

    def forward(self, x, pos, batch):
        e     = tc_knn(pos, pos, self.k, batch, batch)
        c, nb = e[0], e[1]
        dp  = pos[c] - pos[nb]
        pb  = self.pb(dp)
        rel = (self.q(x)[c] - self.kk(x)[nb]) * self.pm(dp) + pb
        wt  = scatter_softmax(self.w(rel), c, dim=0)
        vg  = (self.v(x)[nb] + pb).view(-1, self.g, self.gc)
        out = scatter_add(vg * wt.unsqueeze(-1), c, dim=0, dim_size=x.size(0))
        return out.view(-1, self.g * self.gc)


class Block(nn.Module):
    def __init__(self, ch, g=6, k=16):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.attn = GVA(ch, g, k)
        self.mlp  = nn.Sequential(nn.Linear(ch, ch * 2), nn.ReLU(), nn.Linear(ch * 2, ch))

    def forward(self, x, pos, batch):
        x = x + self.attn(self.n1(x), pos, batch)
        return x + self.mlp(self.n2(x))


class GridPool(nn.Module):
    def __init__(self, i, o, grid):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.Linear(i, o), nn.ReLU())

    def forward(self, x, pos, batch):
        vox = torch.floor(pos / self.grid).long()
        key = torch.cat([batch.unsqueeze(1), vox], 1)
        _, cl = torch.unique(key, dim=0, return_inverse=True)
        xp, _ = scatter_max(self.proj(x), cl, dim=0)
        return xp, scatter_mean(pos, cl, dim=0), scatter_max(batch, cl, dim=0)[0], cl


class PTv2Seg(nn.Module):
    def __init__(self, nc, in_channels, k=16, dims=(48, 96, 192), g=6):
        super().__init__()
        d = dims
        self.embed = nn.Sequential(nn.Linear(in_channels, d[0]), nn.ReLU(),
                                   nn.Linear(d[0], d[0]))
        self.e1 = Block(d[0], g, k); self.p1 = GridPool(d[0], d[1], 0.08)
        self.e2 = Block(d[1], g, k); self.p2 = GridPool(d[1], d[2], 0.16)
        self.e3 = Block(d[2], g, k)
        self.u2 = nn.Sequential(nn.Linear(d[2] + d[1], d[1]), nn.ReLU()); self.d2 = Block(d[1], g, k)
        self.u1 = nn.Sequential(nn.Linear(d[1] + d[0], d[0]), nn.ReLU()); self.d1 = Block(d[0], g, k)
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        h0 = self.e1(self.embed(flat), p0, b0)
        h1, p1, b1, c1 = self.p1(h0, p0, b0); h1 = self.e2(h1, p1, b1)
        h2, p2, b2, c2 = self.p2(h1, p1, b1); h2 = self.e3(h2, p2, b2)
        u1 = self.d2(self.u2(torch.cat([h1, h2[c2]], 1)), p1, b1)
        u0 = self.d1(self.u1(torch.cat([h0, u1[c1]], 1)), p0, b0)
        return self.head(u0).view(B, N, -1).permute(0, 2, 1)


def _fibonacci_kernel(K, radius):
    import math
    pts = [[0.0, 0.0, 0.0]]
    n = K - 1
    phi = (1 + 5 ** 0.5) / 2
    for i in range(n):
        z  = 1 - 2 * (i + 0.5) / n
        r  = (1 - z * z) ** 0.5
        th = 2 * math.pi * i / phi
        pts.append([r * math.cos(th), r * math.sin(th), z])
    kp = torch.tensor(pts, dtype=torch.float32)
    kp[1:] *= radius
    return kp


class KPConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, radius, kernel_points, neighbors):
        super().__init__()
        self.k = neighbors
        self.register_buffer("kp", _fibonacci_kernel(kernel_points, radius * 0.66))
        self.sigma  = radius * 0.3
        self.weight = nn.Parameter(
            torch.randn(kernel_points, in_dim, out_dim) * (2.0 / (in_dim * kernel_points)) ** 0.5)

    def forward(self, x, pos, batch):
        e = tc_knn(pos, pos, self.k, batch, batch)
        c, nb = e[0], e[1]
        rel  = pos[nb] - pos[c]
        d    = (rel.unsqueeze(1) - self.kp.unsqueeze(0)).norm(dim=-1)
        corr = torch.clamp(1.0 - d / self.sigma, min=0.0)
        denom = scatter_add(corr.sum(1, keepdim=True), c, dim=0,
                            dim_size=x.size(0)).clamp(min=1e-6)
        out = x.new_zeros(x.size(0), self.weight.size(2))
        xnb = x[nb]
        for k in range(self.kp.size(0)):
            hk  = scatter_add(corr[:, k:k+1] * xnb, c, dim=0, dim_size=x.size(0))
            out = out + hk @ self.weight[k]
        return out / denom


class KPBlock(nn.Module):
    def __init__(self, ch, radius, kernel_points, neighbors):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.conv = KPConvLayer(ch, ch, radius, kernel_points, neighbors)
        self.mlp  = nn.Sequential(nn.Linear(ch, ch * 2), nn.ReLU(), nn.Linear(ch * 2, ch))

    def forward(self, x, pos, batch):
        x = x + self.conv(self.n1(x), pos, batch)
        return x + self.mlp(self.n2(x))


class KPConvSeg(nn.Module):
    def __init__(self, nc, in_channels, kernel_points=15, neighbors=16,
                radius=0.06, dims=(48, 96, 192)):
        super().__init__()
        d = dims
        self.embed = nn.Sequential(nn.Linear(in_channels, d[0]), nn.ReLU(),
                                   nn.Linear(d[0], d[0]))
        self.e1 = KPBlock(d[0], radius, kernel_points, neighbors)
        self.p1 = GridPool(d[0], d[1], 0.08)
        self.e2 = KPBlock(d[1], radius * 2, kernel_points, neighbors)
        self.p2 = GridPool(d[1], d[2], 0.16)
        self.e3 = KPBlock(d[2], radius * 4, kernel_points, neighbors)
        self.u2 = nn.Sequential(nn.Linear(d[2] + d[1], d[1]), nn.ReLU())
        self.d2 = KPBlock(d[1], radius * 2, kernel_points, neighbors)
        self.u1 = nn.Sequential(nn.Linear(d[1] + d[0], d[0]), nn.ReLU())
        self.d1 = KPBlock(d[0], radius, kernel_points, neighbors)
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        h0 = self.e1(self.embed(flat), p0, b0)
        h1, p1, b1, c1 = self.p1(h0, p0, b0); h1 = self.e2(h1, p1, b1)
        h2, p2, b2, c2 = self.p2(h1, p1, b1); h2 = self.e3(h2, p2, b2)
        u1 = self.d2(self.u2(torch.cat([h1, h2[c2]], 1)), p1, b1)
        u0 = self.d1(self.u1(torch.cat([h0, u1[c1]], 1)), p0, b0)
        return self.head(u0).view(B, N, -1).permute(0, 2, 1)


ARCH_REGISTRY = {"ptv2": PTv2Seg, "kpconv": KPConvSeg}


def load_ensemble_member(arch_name, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    in_ch = ckpt.get("in_channels", 7)
    nc    = int(ckpt.get("num_classes", 2))
    n_pts = int(ckpt.get("num_points", 16384))
    flags = {k: bool(ckpt.get(k, False)) for k in
            ("use_linearity", "use_planarity", "use_sphericity",
             "use_verticality", "use_eigenentropy", "use_rgb")}

    if arch_name == "ptv2":
        model = PTv2Seg(nc=nc, in_channels=in_ch)
    elif arch_name == "kpconv":
        kp_k   = ckpt.get("kp_kernel_points", 15)
        kp_nb  = ckpt.get("kp_neighbors", 16)
        kp_r   = ckpt.get("kp_radius", 0.06)
        kp_dim = tuple(ckpt.get("kp_dims", (48, 96, 192)))
        model = KPConvSeg(nc=nc, in_channels=in_ch, kernel_points=kp_k,
                          neighbors=kp_nb, radius=kp_r, dims=kp_dim)
    else:
        raise ValueError(f"Unknown architecture {arch_name!r}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    miou = ckpt.get("best_val_miou", None)
    print(f"[{arch_name}] loaded {path} | in_channels={in_ch} | "
         f"{'val mIoU=' + f'{miou:.4f}' if miou is not None else ''}")
    return model, n_pts, in_ch, nc, flags


def load_pointcloud(path, return_rgb=False):
    las = laspy.read(path)
    pts = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                           np.asarray(las.z)]).astype(np.float64)
    labels = None
    for key in ("classification", "label", "labels", "class"):
        if key in las.point_format.dimension_names:
            labels = np.asarray(getattr(las, key), dtype=np.int64)
            break
    rgb = None
    if return_rgb:
        dims = las.point_format.dimension_names
        if all(k in dims for k in ("red", "green", "blue")):
            rgb = np.column_stack([np.asarray(las.red), np.asarray(las.green),
                                   np.asarray(las.blue)]).astype(np.float32)
            if rgb.max() > 255: rgb /= 65535.0
            elif rgb.max() > 1: rgb /= 255.0
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        if labels is not None: labels = labels[finite]
        if rgb is not None: rgb = rgb[finite]
    if return_rgb:
        return pts, rgb, labels
    return pts, labels


def _voxel_keep(pts, voxel):
    vox = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((cnt.size, 3), np.float64); np.add.at(sums, inv, pts)
    d2 = ((pts - (sums / cnt[:, None])[inv]) ** 2).sum(1)
    order = np.lexsort((d2, inv))
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return np.sort(order[first])


def _sor_keep(pts, k=16, std_ratio=2.0):
    n = len(pts)
    if n <= k + 1:
        return np.ones(n, bool)
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_dist = d[:, 1:].mean(axis=1)
    thr = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= thr


def _cluster_keep(pts, eps, min_samples=8, min_cluster_size=100):
    from sklearn.cluster import DBSCAN
    if len(pts) < min_samples:
        return np.ones(len(pts), bool)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(pts)
    keep = np.zeros(len(pts), bool)
    for lab in np.unique(labels):
        if lab == -1: continue
        idx = np.flatnonzero(labels == lab)
        if len(idx) >= min_cluster_size: keep[idx] = True
    return keep


def _geom_features(points, radius, min_neighbors=8, want_linear=True,
                   want_planar=False, want_sphere=False, want_vert=False,
                   want_entropy=False):
    n = len(points)
    lin  = np.zeros(n, np.float32) if want_linear else None
    pla  = np.zeros(n, np.float32) if want_planar else None
    sph  = np.zeros(n, np.float32) if want_sphere else None
    vert = np.zeros(n, np.float32) if want_vert else None
    ent  = np.zeros(n, np.float32) if want_entropy else None
    if not (want_linear or want_planar or want_sphere or want_vert or want_entropy):
        return lin, pla, sph, vert, ent
    need_evecs = want_vert
    tree = cKDTree(points)
    lists = tree.query_ball_point(points, r=radius, workers=-1)
    for i, nbrs in enumerate(lists):
        if len(nbrs) < min_neighbors: continue
        nb = points[nbrs]
        c = nb - nb.mean(0)
        cov = (c.T @ c) / len(nbrs)
        if need_evecs:
            evals, evecs = np.linalg.eigh(cov)
            l3, l2, l1 = evals
        else:
            l3, l2, l1 = np.linalg.eigvalsh(cov)
        l1c = max(l1, 1e-9)
        if want_linear:  lin[i] = (l1 - l2) / l1c
        if want_planar:  pla[i] = (l2 - l3) / l1c
        if want_sphere:  sph[i] = l3 / l1c
        if want_vert:
            vert[i] = 1.0 - abs(evecs[:, 0][2])
        if want_entropy:
            s = l1 + l2 + l3
            if s > 1e-9:
                p = np.clip(np.array([l1, l2, l3]) / s, 1e-12, None)
                ent[i] = -(p * np.log(p)).sum()
    return lin, pla, sph, vert, ent


def make_features(points, rgb, in_channels, flags):
    center = points.mean(axis=0, keepdims=True)
    scale  = max(np.linalg.norm(points - center, axis=1).max(), 1e-9)
    norm_xyz = ((points - center) / scale).astype(np.float32)
    z = points[:, 2]
    height = ((z - z.min()) / max(z.max() - z.min(), 1e-6)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=PREP["normal_radius"], max_nn=PREP["normal_max_nn"]))
    pcd.orient_normals_to_align_with_direction([0., 0., 1.])
    normals = np.asarray(pcd.normals, dtype=np.float32)
    cols = [norm_xyz, height[:, None], normals]

    want_any_geom = any(flags.get(k, False) for k in
                        ("use_linearity", "use_planarity", "use_sphericity",
                         "use_verticality", "use_eigenentropy"))
    if want_any_geom:
        lin, pla, sph, vert, ent = _geom_features(
            points, radius=PREP["geom_radius"],
            want_linear=flags.get("use_linearity", False),
            want_planar=flags.get("use_planarity", False),
            want_sphere=flags.get("use_sphericity", False),
            want_vert=flags.get("use_verticality", False),
            want_entropy=flags.get("use_eigenentropy", False))
        if flags.get("use_linearity"):    cols.append(lin[:, None])
        if flags.get("use_planarity"):    cols.append(pla[:, None])
        if flags.get("use_sphericity"):   cols.append(sph[:, None])
        if flags.get("use_verticality"):  cols.append(vert[:, None])
        if flags.get("use_eigenentropy"): cols.append(ent[:, None])

    if flags.get("use_rgb", False):
        if rgb is None:
            rgb = np.zeros((len(points), 3), np.float32)
        cols.append(rgb.astype(np.float32))

    feat = np.column_stack(cols)
    if feat.shape[1] != in_channels:
        raise ValueError(f"Assembled {feat.shape[1]} channels but expected {in_channels}")
    return feat


@torch.no_grad()
def predict_vote_sum(model, pts, feat, num_points, num_classes, overlap_factor=2):
    """Same kNN-sphere-chunk inference as the notebook's predict_full_cloud,
    but returns the raw per-class VOTE-SUM array instead of the final
    argmax — this is what lets multiple models' votes be combined before
    the final decision (a proper ensemble, not just majority-of-labels)."""
    model.eval()
    n = len(feat)
    N = min(num_points, n)
    model.to(DEVICE)
    B = 8

    if n <= N:
        pad_idx = np.random.RandomState(0).choice(n, N, replace=True)
        x = torch.from_numpy(feat[pad_idx][None].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy().ravel()
        vote_sum = np.zeros((n, num_classes), np.int32)
        np.add.at(vote_sum, (pad_idx, out), 1)
        model.to("cpu")
        return vote_sum

    tree = cKDTree(pts)
    covered = np.zeros(n, bool)
    vote_sum = np.zeros((n, num_classes), np.int32)
    rng = np.random.RandomState(0)
    batch_ids = []

    def flush():
        if not batch_ids: return
        x = torch.from_numpy(feat[np.stack(batch_ids)].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy()
        for ids, o in zip(batch_ids, out):
            np.add.at(vote_sum, (ids, o), 1)
        batch_ids.clear()

    target_visits = max(1, overlap_factor)
    visit_count = np.zeros(n, np.int32)
    max_iters = int(np.ceil(n / N * target_visits)) + 4
    for _ in range(max_iters):
        if (visit_count >= target_visits).all(): break
        least = np.flatnonzero(visit_count == visit_count.min())
        seed = least[rng.randint(len(least))]
        _, idx = tree.query(pts[seed], k=N)
        idx = np.atleast_1d(idx)
        batch_ids.append(idx)
        visit_count[idx] += 1
        covered[idx] = True
        if len(batch_ids) == B:
            flush()
    flush()

    if not covered.all() and covered.any():
        missing = np.flatnonzero(~covered)
        covered_idx = np.flatnonzero(covered)
        _, nn_local = cKDTree(pts[covered_idx]).query(pts[missing], k=1)
        vote_sum[missing] = vote_sum[covered_idx[nn_local]]

    model.to("cpu")
    return vote_sum


def ensemble_predict(path, checkpoints=ENSEMBLE_CHECKPOINTS):
    """Loads every (arch, checkpoint_path) pair, runs each model's own
    preprocessing+feature-computation (each checkpoint's own channel flags
    are honoured independently), sums their vote_sum arrays with EQUAL
    weight, and returns the combined ensemble prediction plus each
    individual model's own prediction (for side-by-side comparison)."""
    pts_ref, rgb_ref, lbl = load_pointcloud(path, return_rgb=True)
    v = PREP["voxel_size"]
    keep = _voxel_keep(pts_ref, v)
    pts_ref, lbl = pts_ref[keep], (lbl[keep] if lbl is not None else None)
    if rgb_ref is not None: rgb_ref = rgb_ref[keep]
    if PREP["sor_enabled"]:
        keep = _sor_keep(pts_ref, k=PREP["sor_k"], std_ratio=PREP["sor_std"])
        pts_ref, lbl = pts_ref[keep], (lbl[keep] if lbl is not None else None)
        if rgb_ref is not None: rgb_ref = rgb_ref[keep]
    if PREP["cluster_filter_enabled"]:
        keep = _cluster_keep(pts_ref, eps=PREP["cluster_eps"],
                             min_cluster_size=PREP["cluster_min_size"])
        pts_ref, lbl = pts_ref[keep], (lbl[keep] if lbl is not None else None)
        if rgb_ref is not None: rgb_ref = rgb_ref[keep]

    n = len(pts_ref)
    combined_vote = None
    individual_preds = {}

    for arch_name, ckpt_path in checkpoints:
        model, n_pts, in_ch, nc, flags = load_ensemble_member(arch_name, ckpt_path)
        feat = make_features(pts_ref, rgb_ref, in_ch, flags)
        vote_sum = predict_vote_sum(model, pts_ref, feat, n_pts, nc)
        if combined_vote is None:
            combined_vote = np.zeros((n, nc), np.float64)
        combined_vote += vote_sum.astype(np.float64) / vote_sum.sum(1, keepdims=True).clip(min=1)
        individual_preds[arch_name] = vote_sum.argmax(axis=1).astype(np.int64)

    ensemble_pred = combined_vote.argmax(axis=1).astype(np.int64)
    return pts_ref, ensemble_pred, individual_preds, lbl


def _iou(true, pred, cls):
    tp = int(((true == cls) & (pred == cls)).sum())
    fp = int(((true != cls) & (pred == cls)).sum())
    fn = int(((true == cls) & (pred != cls)).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


if __name__ == "__main__":
    pts, ens_pred, indiv, lbl = ensemble_predict(LAS_PATH)
    n_tgt = int((ens_pred == TARGET_CLASS).sum())
    print(f"\nEnsemble: {len(pts):,} pts | target={n_tgt:,} "
         f"({n_tgt/len(pts)*100:.1f}%)")

    if lbl is not None and np.unique(lbl).size > 1:
        print(f"\n{'Model':<12}{'IoU (target class)':>20}")
        for name, pred in indiv.items():
            print(f"{name:<12}{_iou(lbl, pred, TARGET_CLASS):>20.4f}")
        print(f"{'ENSEMBLE':<12}{_iou(lbl, ens_pred, TARGET_CLASS):>20.4f}")
