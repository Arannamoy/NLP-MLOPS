# ═════════════════════════════════════════════════════════════════════════════
# Standalone PTv2 inference — single .las file → segmentation → Open3D window
#
# Usage:   python inference_standalone.py [path/to/file.las]
#
# Pipeline is IDENTICAL to PointTransformerV2_segmentation.ipynb:
#   load .las → 7-ch features (real Open3D normals) → kNN-sphere chunk
#   inference with majority vote → visualize (green = target = class 1)
# ═════════════════════════════════════════════════════════════════════════════
import sys, os
import numpy as np
import laspy
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch_cluster import knn as tc_knn
from torch_scatter import scatter_softmax, scatter_add, scatter_max, scatter_mean

MODEL_PATH   = "checkpoints/PointTransformerV2_best.pth"
LAS_PATH     = sys.argv[1] if len(sys.argv) > 1 else "data/test/sample_data_0046.las"
TARGET_CLASS = 1                      # 1 = wood powder, 0 = environment
NUM_CLASSES  = 2
FALLBACK_N   = 8192                   # used only if checkpoint lacks "num_points"
BATCH_SIZE   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── PTv2 architecture (must match training exactly) ──────────────────────────
class GVA(nn.Module):
    "Grouped Vector Attention — PTv2 core attention layer."
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
    def __init__(self, nc=NUM_CLASSES, k=16, dims=(48, 96, 192), g=6):
        super().__init__()
        d = dims
        self.embed = nn.Sequential(nn.Linear(7, d[0]), nn.ReLU(), nn.Linear(d[0], d[0]))
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


# ── Model loading — best.pth is a CHECKPOINT DICT, not a model object ────────
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{path} does not look like a training checkpoint "
                         f"(expected a dict with 'model_state').")
    model = PTv2Seg(nc=int(ckpt.get("num_classes", NUM_CLASSES)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_pts = int(ckpt.get("num_points", FALLBACK_N))
    miou  = ckpt.get("best_val_miou", None)
    print(f"Loaded {path}"
          + (f" | best val mIoU {miou:.4f}" if miou is not None else "")
          + f" | chunk size {n_pts:,}")
    return model, n_pts


# ── Data loading + features (identical to the notebook) ──────────────────────
def load_pointcloud(path):
    las = laspy.read(path)
    pts = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                           np.asarray(las.z)]).astype(np.float64)
    labels = None
    for key in ("classification", "label", "labels", "class"):
        if key in las.point_format.dimension_names:
            labels = np.asarray(getattr(las, key), dtype=np.int64)
            break
    return pts, labels


def make_features(points):
    center = points.mean(axis=0, keepdims=True)
    scale  = max(np.linalg.norm(points - center, axis=1).max(), 1e-9)
    norm_xyz = ((points - center) / scale).astype(np.float32)

    z = points[:, 2]
    height = ((z - z.min()) / max(z.max() - z.min(), 1e-6)).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    pcd.orient_normals_to_align_with_direction([0., 0., 1.])
    normals = np.asarray(pcd.normals, dtype=np.float32)

    return np.column_stack([norm_xyz, height[:, None], normals])


# ── Full-cloud inference: kNN-sphere chunks + majority vote ──────────────────
@torch.no_grad()
def predict_full_cloud(model, path, num_points):
    pts, lbl = load_pointcloud(path)
    lbl  = np.zeros(len(pts), np.int64) if lbl is None else \
           np.clip(lbl, 0, NUM_CLASSES - 1).astype(np.int64)
    print(f"{os.path.basename(path)}: {len(pts):,} points — computing features…")
    feat = make_features(pts)

    n = len(feat)
    N = min(num_points, n)
    model.to(DEVICE)

    tree      = cKDTree(pts)
    covered   = np.zeros(n, bool)
    vote_tgt  = np.zeros(n, np.int64)
    vote_cnt  = np.zeros(n, np.int64)
    rng       = np.random.RandomState(0)
    batch_ids = []
    n_chunks  = 0

    def flush():
        nonlocal n_chunks
        if not batch_ids:
            return
        x   = torch.from_numpy(feat[np.stack(batch_ids)].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy()
        for ids, o in zip(batch_ids, out):
            np.add.at(vote_tgt, ids, o)
            np.add.at(vote_cnt, ids, 1)
        n_chunks += len(batch_ids)
        batch_ids.clear()

    while not covered.all():
        uncovered = np.flatnonzero(~covered)
        seed      = uncovered[rng.randint(len(uncovered))]
        _, idx    = tree.query(pts[seed], k=N)
        batch_ids.append(np.atleast_1d(idx))
        covered[batch_ids[-1]] = True
        if len(batch_ids) == BATCH_SIZE:
            flush()
            print(f"\r  inference… {covered.sum():,}/{n:,} pts "
                  f"({n_chunks} chunks)", end="")
    flush()
    print(f"\r  inference done: {n_chunks} chunks on {DEVICE}          ")
    model.to("cpu")
    return pts, (vote_tgt * 2 >= vote_cnt).astype(np.int64), lbl


# ── Visualization — green = TARGET (class 1), red = others ───────────────────
def visualize_segmentation(points, predictions, title="Segmentation"):
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[predictions == TARGET_CLASS] = [0.0, 0.8, 0.0]   # green = target
    colors[predictions != TARGET_CLASS] = [0.8, 0.0, 0.0]   # red   = others

    pcd = o3d.geometry.PointCloud()
    center = points.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector((points - center).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    span = float((points.max(0) - points.min(0)).max()) * 0.5
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=span)

    n_target = int((predictions == TARGET_CLASS).sum())   # class 1 = target!
    n_other  = len(predictions) - n_target
    pct      = n_target / max(len(predictions), 1) * 100
    full_title = (f"{title}  |  green(target)={n_target:,} ({pct:.1f}%)  "
                  f"red(others)={n_other:,}")
    print(full_title)
    o3d.visualization.draw_geometries([pcd, axes], window_name=full_title,
                                      width=1280, height=800)


if __name__ == "__main__":
    model, n_pts = load_model(MODEL_PATH)
    pts, preds, lbl = predict_full_cloud(model, LAS_PATH, n_pts)

    # if the file has ground-truth labels, print the exact mIoU — no guessing
    if lbl.any() or (lbl == 0).sum() < len(lbl):
        inter_u = []
        for c in range(NUM_CLASSES):
            tp = int(((lbl == c) & (preds == c)).sum())
            fp = int(((lbl != c) & (preds == c)).sum())
            fn = int(((lbl == c) & (preds != c)).sum())
            if tp + fp + fn:
                inter_u.append(tp / (tp + fp + fn))
        if inter_u:
            print(f"GT labels found → per-file mIoU = {float(np.mean(inter_u)):.4f}")

    visualize_segmentation(pts, preds, title="INFERENCE | PointTransformerV2 | "
                           + os.path.splitext(os.path.basename(LAS_PATH))[0])