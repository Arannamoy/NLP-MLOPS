
# %%

import os, glob, glob, copy, random, logging
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import laspy
import joblib
import mlflow
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Running on: {DEVICE}")


# %%
CONFIG = {
    "train_dir"      : "../data/train_v1",
    "test_dir"       : "../data/test",
    "checkpoint_dir" : "../checkpoints/PointCloudDataset_SlidingWindow_9_channel",
    "num_classes"    : 2,
    "target_class"   : 1,
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "seed"           : 42,
    "epochs"         : 150,
    "patience"       : 40,
    "batch_size"     : 8,
    "num_points"     : 16384,
    "chunks_per_cloud": 100,
    "lr"             : 2e-4,
    "grad_accum"     : 2,
    "sched_factor"   : 0.5,
    "sched_patience" : 10,
    "min_lr"         : 1e-6,
    "voxel_size"     : 0.01,
    "normal_radius"  : 0.05,
    "normal_max_nn"  : 30,
    "sor_enabled"    : True,
    "sor_k"          : 16,
    "sor_std"        : 2.0,
    "cluster_filter_enabled": True,
    "cluster_eps"    : 0.03,
    "cluster_min_size": 100,
    "aug_jitter"     : 0.01,
    "aug_scale"      : 0.1,
    "aug_dropout"    : 0.10,
    "use_linearity"  : True,
    "use_planarity"  : True,
    "use_sphericity" : False,
    "geom_radius"    : 0.06,
    "use_rgb"        : False,
    "in_channels"    : 7,
    "num_workers"    : 4,
    "use_amp"        : True,
    "class_weights"  : None,
    "max_vis_files"  : 3,
    "mlflow_uri"     : "http://localhost:5000",
    "mlflow_experiment": "PointCloudDataset_SlidingWindow_9_channel",
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
CONFIG["in_channels"] = (7 + (1 if CONFIG["use_linearity"] else 0)
                           + (1 if CONFIG["use_planarity"] else 0)
                           + (1 if CONFIG["use_sphericity"] else 0)
                           + (3 if CONFIG["use_rgb"] else 0))
NUM_CLASSES = CONFIG["num_classes"]

def compute_class_weights(files, cap=10.0):
    """Inverse-frequency class weights from a sample of labelled files.
    FIX: wood-powder (class 1) is under-represented in several scans;
    call this once after TRAIN_FILES is known and assign the result to
    CONFIG["class_weights"] before training. `cap` bounds the max weight
    ratio so a rare-but-noisy class doesn't destabilise training."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in files:
        _, lbl = load_pointcloud(f)
        if lbl is None:
            continue
        lbl = np.clip(lbl, 0, NUM_CLASSES - 1)
        counts += np.bincount(lbl, minlength=NUM_CLASSES)
    if counts.min() == 0:
        return None
    freq = counts / counts.sum()
    w = 1.0 / np.clip(freq, 1e-6, None)
    w = w / w.mean()
    w = np.clip(w, 1.0 / cap, cap)
    return w.tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])


# %%
def load_pointcloud(path, return_rgb=False):
    """Read a .las/.laz file. Returns (points[N,3], labels[N] or None),
    or (points, rgb[N,3] float32 0-1 or None, labels) when return_rgb=True.
    RGB in .las is 16-bit (0-65535); normalised here to 0-1."""
    las = laspy.read(path)
    pts = np.column_stack([np.asarray(las.x),
                           np.asarray(las.y),
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
            rgb = np.column_stack([np.asarray(las.red),
                                   np.asarray(las.green),
                                   np.asarray(las.blue)]).astype(np.float32)
            if rgb.max() > 255:
                rgb /= 65535.0
            elif rgb.max() > 1:
                rgb /= 255.0

    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        n_bad = int((~finite).sum())
        log.warning(f"{os.path.basename(path)}: dropping {n_bad} non-finite points")
        pts = pts[finite]
        if labels is not None:
            labels = labels[finite]
        if rgb is not None:
            rgb = rgb[finite]
    if labels is not None:
        assert len(labels) == len(pts), \
            f"{os.path.basename(path)}: label/point count mismatch"
    if return_rgb:
        return pts, rgb, labels
    return pts, labels

def list_files(folder):
    """Return sorted list of .las/.laz files in a folder."""
    files = []
    for ext in (".las", ".laz"):
        files += glob.glob(os.path.join(folder, "*" + ext))
    return sorted(files)

all_files = list_files(CONFIG["train_dir"])
assert all_files, f"No .las files found in {CONFIG['train_dir']}"

rng   = np.random.RandomState(CONFIG["seed"])
order = rng.permutation(len(all_files))
n_val  = max(1, int(len(all_files) * CONFIG["val_ratio"]))
n_test = max(1, int(len(all_files) * CONFIG["test_ratio"]))

VAL_FILES   = [all_files[i] for i in order[:n_val]]
TEST_FILES  = [all_files[i] for i in order[n_val : n_val + n_test]]
TRAIN_FILES = [all_files[i] for i in order[n_val + n_test :]]
INFER_FILES = list_files(CONFIG["test_dir"])

log.info(f"train={len(TRAIN_FILES)}  val={len(VAL_FILES)}  "
         f"test={len(TEST_FILES)}  inference={len(INFER_FILES)}")


# %%
def _geom_features(points, radius=0.06, min_neighbors=8,
                   want_linear=True, want_planar=False, want_sphere=False):
    """Per-point PCA shape descriptors (Weinmann et al.), FIXED-RADIUS
    neighbourhood, computed from the SAME local covariance eigenvalues
    (λ1≥λ2≥λ3) so all three come at ~one KDTree-query cost:
        linearity  = (λ1-λ2)/λ1  — thin/elongated (fin/comb scan-artifact)
        planarity  = (λ2-λ3)/λ1  — flat surface (wall/floor/pile top)
        sphericity = λ3/λ1       — blob-like / isotropic noise cluster
    Verified: flat surface linearity≈0.3-0.4, thin fin≈0.9+. This lets the
    model SEE the anisotropic fin geometry directly — with only
    xyz+height+normals the artifact is nearly indistinguishable from real
    surface, which is why both earlier attempts (train with/without the
    artifact present) failed to fix it on their own."""
    from scipy.spatial import cKDTree as _GeomTree
    n = len(points)
    lin = np.zeros(n, np.float32) if want_linear else None
    pla = np.zeros(n, np.float32) if want_planar else None
    sph = np.zeros(n, np.float32) if want_sphere else None
    if not (want_linear or want_planar or want_sphere):
        return lin, pla, sph

    tree = _GeomTree(points)
    lists = tree.query_ball_point(points, r=radius, workers=-1)
    for i, nbrs in enumerate(lists):
        if len(nbrs) < min_neighbors:
            continue
        nb = points[nbrs]
        c = nb - nb.mean(0)
        cov = (c.T @ c) / len(nbrs)
        l3, l2, l1 = np.linalg.eigvalsh(cov)
        l1 = max(l1, 1e-9)
        if want_linear:  lin[i] = (l1 - l2) / l1
        if want_planar:  pla[i] = (l2 - l3) / l1
        if want_sphere:  sph[i] = l3 / l1
    return lin, pla, sph


def make_features(points, rgb=None):
    """Feature assembly — channel layout (fixed order):
        [0:3]  normalized xyz
        [3]    normalized height
        [4:7]  surface normals            (rotated in augmentation)
        [7]    linearity   — if CONFIG["use_linearity"]  (rotation-invariant)
        [7/8+] rgb (3ch)   — if CONFIG["use_rgb"]        (rotation-invariant)
    Returns float32 (N, CONFIG["in_channels"])."""
    center = points.mean(axis=0, keepdims=True)
    scale  = max(np.linalg.norm(points - center, axis=1).max(), 1e-9)
    norm_xyz = ((points - center) / scale).astype(np.float32)

    z = points[:, 2]
    height = ((z - z.min()) / max(z.max() - z.min(), 1e-6)).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=CONFIG.get("normal_radius", 0.05),
        max_nn=CONFIG.get("normal_max_nn", 30)))
    pcd.orient_normals_to_align_with_direction([0., 0., 1.])
    normals = np.asarray(pcd.normals, dtype=np.float32)

    cols = [norm_xyz, height[:, None], normals]

    want_lin = CONFIG.get("use_linearity", False)
    want_pla = CONFIG.get("use_planarity", False)
    want_sph = CONFIG.get("use_sphericity", False)
    if want_lin or want_pla or want_sph:
        lin, pla, sph = _geom_features(
            points, radius=CONFIG.get("geom_radius", 0.06),
            want_linear=want_lin, want_planar=want_pla, want_sphere=want_sph)
        if want_lin: cols.append(lin[:, None])
        if want_pla: cols.append(pla[:, None])
        if want_sph: cols.append(sph[:, None])

    if CONFIG.get("use_rgb", False):
        if rgb is None:
            log.warning("use_rgb=True but file has no RGB — filling zeros")
            rgb = np.zeros((len(points), 3), np.float32)
        cols.append(rgb.astype(np.float32))

    feat = np.column_stack(cols)
    assert feat.shape[1] == CONFIG["in_channels"], \
        f"feature dims {feat.shape[1]} != CONFIG in_channels {CONFIG['in_channels']}"
    return feat

def _voxel_keep(pts, voxel):
    """O(N) voxel downsample — one representative point per voxel (closest to
    centroid). Deterministic; keeps labels aligned via returned indices.
    ISSUE 3/5: reduces dense-scan redundancy so duplicate points don't bias
    normal statistics, and speeds up per-file feature computation."""
    vox = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((cnt.size, 3), np.float64); np.add.at(sums, inv, pts)
    d2 = ((pts - (sums / cnt[:, None])[inv]) ** 2).sum(1)
    order = np.lexsort((d2, inv))
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return np.sort(order[first])

def _sor_keep(pts, k=16, std_ratio=2.0):
    """Statistical Outlier Removal — returns a boolean keep-mask.
    FIX: this was previously only applied inside vol_TIN() for volume
    estimation, NEVER inside get_features(). That meant training data (if
    cleaned offline before being placed in data/train) and raw inference
    files saw DIFFERENT distributions — the model never learned what stray
    LiDAR noise looks like, so at inference it had to guess and often
    mis-classified floating outlier clusters as target. Applying SOR here
    makes train and inference preprocessing IDENTICAL.
    A point is dropped if its mean distance to its k nearest neighbours is
    more than `std_ratio` standard deviations above the mean-of-means."""
    from scipy.spatial import cKDTree as _SORTree
    n = len(pts)
    if n <= k + 1:
        return np.ones(n, bool)
    d, _ = _SORTree(pts).query(pts, k=k + 1)
    mean_dist = d[:, 1:].mean(axis=1)
    thr = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= thr


def _cluster_keep(pts, eps, min_samples=8, min_cluster_size=100):
    """DBSCAN connected-component filter — catches CLUSTERED noise blobs
    (e.g. floating dust/mist/multi-path plumes) that plain distance-based
    SOR MISSES. SOR only flags isolated points: if outlier points are
    tightly packed among THEMSELVES (each other's nearest neighbour), their
    local density looks "normal" to SOR and they survive — measured 47%
    catch-rate on a clustered blob vs 99% on scattered noise. This function
    instead keeps only points belonging to connected components with at
    least `min_cluster_size` points; small floating clusters (real
    structures — floor/wall/pile — are always large) are dropped along with
    DBSCAN-noise (isolated points, label -1)."""
    from sklearn.cluster import DBSCAN as _DBSCAN
    if len(pts) < min_samples:
        return np.ones(len(pts), bool)
    labels = _DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(pts)
    keep = np.zeros(len(pts), bool)
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.flatnonzero(labels == lab)
        if len(idx) >= min_cluster_size:
            keep[idx] = True
    return keep

FEATURE_CACHE = {}

def _prep_key(path):
    return (path,
            round(float(CONFIG.get("voxel_size", 0)), 6),
            round(float(CONFIG.get("normal_radius", 0.05)), 6),
            int(CONFIG.get("normal_max_nn", 30)),
            bool(CONFIG.get("sor_enabled", True)),
            round(float(CONFIG.get("sor_std", 2.0)), 3),
            int(CONFIG.get("sor_k", 16)),
            bool(CONFIG.get("cluster_filter_enabled", True)),
            round(float(CONFIG.get("cluster_eps", 0.03)), 4),
            int(CONFIG.get("cluster_min_size", 100)),
            bool(CONFIG.get("use_linearity", False)),
            bool(CONFIG.get("use_planarity", False)),
            bool(CONFIG.get("use_sphericity", False)),
            round(float(CONFIG.get("geom_radius", 0.06)), 4),
            bool(CONFIG.get("use_rgb", False)))

def get_features(path):
    """Return (pts[N,3], features[N,7], labels[N]), cached per (file, config).
    Voxel step is for normal quality + speed, NOT for coverage — coverage is
    handled by grid chunking in the Dataset."""
    key = _prep_key(path)
    if key not in FEATURE_CACHE:
        pts, rgb, lbl = load_pointcloud(path, return_rgb=True)
        lbl = np.zeros(len(pts), dtype=np.int64) if lbl is None else lbl
        lbl = np.clip(lbl, 0, NUM_CLASSES - 1).astype(np.int64)
        v = CONFIG.get("voxel_size", 0)
        if v and v > 0 and len(pts) > 0:
            keep = _voxel_keep(pts, v)
            pts, lbl = pts[keep], lbl[keep]
            if rgb is not None: rgb = rgb[keep]

        if CONFIG.get("sor_enabled", True) and len(pts) > 0:
            keep = _sor_keep(pts, k=CONFIG.get("sor_k", 16),
                             std_ratio=CONFIG.get("sor_std", 2.0))
            n_removed = int((~keep).sum())
            if n_removed > 0:
                log.info(f"  SOR: removed {n_removed} outlier pts "
                         f"({os.path.basename(path)})")
            pts, lbl = pts[keep], lbl[keep]
            if rgb is not None: rgb = rgb[keep]

        if CONFIG.get("cluster_filter_enabled", True) and len(pts) > 0:
            keep = _cluster_keep(pts,
                                 eps=CONFIG.get("cluster_eps", 0.03),
                                 min_samples=8,
                                 min_cluster_size=CONFIG.get("cluster_min_size", 100))
            n_removed = int((~keep).sum())
            if n_removed > 0:
                log.info(f"  Cluster-filter: removed {n_removed} pts in small "
                         f"floating clusters ({os.path.basename(path)})")
            pts, lbl = pts[keep], lbl[keep]
            if rgb is not None: rgb = rgb[keep]

        feat = make_features(pts, rgb)
        FEATURE_CACHE[key] = (pts, feat, lbl)
    return FEATURE_CACHE[key]

log.info("Pre-computing features for train + val files (one-time cost)...")
for f in tqdm(TRAIN_FILES + VAL_FILES, desc="features"):
    get_features(f)
log.info("Done.")


# %%
def compute_miou(true_labels, pred_labels, num_classes):
    """Compute mean Intersection-over-Union across all classes."""
    ious = []
    for c in range(num_classes):
        tp = int(((true_labels == c) & (pred_labels == c)).sum())
        fp = int(((true_labels != c) & (pred_labels == c)).sum())
        fn = int(((true_labels == c) & (pred_labels != c)).sum())
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else 0.0


def compute_dice(true_labels, pred_labels, num_classes):
    """Compute mean Dice Score across all classes."""
    dices = []
    for c in range(num_classes):
        tp = int(((true_labels == c) & (pred_labels == c)).sum())
        fp = int(((true_labels != c) & (pred_labels == c)).sum())
        fn = int(((true_labels == c) & (pred_labels != c)).sum())
        if tp + fp + fn > 0:
            dices.append((2 * tp) / (2 * tp + fp + fn))
    return float(np.mean(dices)) if dices else 0.0

def per_file_report(names, true_list, pred_list, tag="", target_class=1):
    """Print one row per file: IoU, Dice, Precision, Recall, TP, FP, FN, TN,
    GT Wood, Pred Wood — for the target (wood-powder) class. Works for
    train/val/test (labelled) or inference (true_list entries can be None,
    in which case only 'Pred Wood' is shown for that file)."""
    header = (f"{'File':<22}{'IoU':>7}{'Dice':>7}{'Prec':>7}{'Rec':>7}"
             f"{'TP':>9}{'FP':>9}{'FN':>9}{'TN':>9}{'GT_Wood':>10}{'Pred_Wood':>11}")
    print(f"\n{'='*len(header)}")
    print(f"PER-FILE SEGMENTATION REPORT{' — ' + tag if tag else ''}")
    print('='*len(header))
    print(header)
    print('-'*len(header))

    ious, dices, precs, recs = [], [], [], []
    for name, true, pred in zip(names, true_list, pred_list):
        pred_wood = int((pred == target_class).sum())
        if true is None:
            print(f"{name:<22}{'—':>7}{'—':>7}{'—':>7}{'—':>7}"
                 f"{'—':>9}{'—':>9}{'—':>9}{'—':>9}{'—':>10}{pred_wood:>11,}")
            continue

        tp = int(((true == target_class) & (pred == target_class)).sum())
        fp = int(((true != target_class) & (pred == target_class)).sum())
        fn = int(((true == target_class) & (pred != target_class)).sum())
        tn = int(((true != target_class) & (pred != target_class)).sum())
        gt_wood = int((true == target_class).sum())

        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ious.append(iou); dices.append(dice); precs.append(prec); recs.append(rec)

        print(f"{name:<22}{iou:7.3f}{dice:7.3f}{prec:7.3f}{rec:7.3f}"
             f"{tp:9,}{fp:9,}{fn:9,}{tn:9,}{gt_wood:10,}{pred_wood:11,}")

    if ious:
        print('-'*len(header))
        print(f"{'MEAN':<22}{np.mean(ious):7.3f}{np.mean(dices):7.3f}"
             f"{np.mean(precs):7.3f}{np.mean(recs):7.3f}")
    print('='*len(header))

# %%
try:
    mlflow.set_tracking_uri(CONFIG["mlflow_uri"])
    mlflow.set_experiment(CONFIG["mlflow_experiment"])
    MLFLOW_OK = True
    log.info(f"MLflow tracking: {CONFIG['mlflow_uri']}")
except Exception as e:
    MLFLOW_OK = False
    log.warning(f"MLflow not available ({e}) — training continues without logging")


# %% [markdown]
# ## Model Architecture

# %%
MODEL_NAME = "PointTransformerV2"

from torch_cluster import knn as tc_knn
from torch_scatter import scatter_softmax, scatter_add, scatter_max, scatter_mean

class GVA(nn.Module):
    "Grouped Vector Attention — PTv2 core attention layer."
    def __init__(self, ch, g=6, k=16):
        super().__init__()
        assert ch % g == 0
        self.k, self.g, self.gc = k, g, ch // g
        self.q  = nn.Linear(ch, ch); self.kk = nn.Linear(ch, ch); self.v = nn.Linear(ch, ch)
        self.pm = nn.Sequential(nn.Linear(3,ch), nn.ReLU(), nn.Linear(ch,ch))
        self.pb = nn.Sequential(nn.Linear(3,ch), nn.ReLU(), nn.Linear(ch,ch))
        self.w  = nn.Sequential(nn.Linear(ch,ch), nn.ReLU(), nn.Linear(ch,g))
    def forward(self, x, pos, batch):
        e    = tc_knn(pos, pos, self.k, batch, batch)
        c, nb = e[0], e[1]
        dp   = pos[c] - pos[nb]
        pb   = self.pb(dp)
        rel  = (self.q(x)[c] - self.kk(x)[nb]) * self.pm(dp) + pb
        wt   = scatter_softmax(self.w(rel), c, dim=0)
        vg   = (self.v(x)[nb] + pb).view(-1, self.g, self.gc)
        out  = scatter_add(vg * wt.unsqueeze(-1), c, dim=0, dim_size=x.size(0))
        return out.view(-1, self.g * self.gc)

class Block(nn.Module):
    def __init__(self, ch, g=6, k=16):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.attn = GVA(ch, g, k)
        self.mlp  = nn.Sequential(nn.Linear(ch,ch*2), nn.ReLU(), nn.Linear(ch*2,ch))
    def forward(self, x, pos, batch):
        x = x + self.attn(self.n1(x), pos, batch)
        return x + self.mlp(self.n2(x))

class GridPool(nn.Module):
    def __init__(self, i, o, grid):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.Linear(i,o), nn.ReLU())
    def forward(self, x, pos, batch):
        vox = torch.floor(pos/self.grid).long()
        key = torch.cat([batch.unsqueeze(1), vox], 1)
        _, cl = torch.unique(key, dim=0, return_inverse=True)
        xp, _ = scatter_max(self.proj(x), cl, dim=0)
        return xp, scatter_mean(pos,cl,dim=0), scatter_max(batch,cl,dim=0)[0], cl

class PTv2Seg(nn.Module):
    def __init__(self, nc=NUM_CLASSES, k=16, dims=(48,96,192), g=6):
        super().__init__()
        d = dims
        self.embed = nn.Sequential(
            nn.Linear(CONFIG["in_channels"], d[0]), nn.ReLU(),
            nn.Linear(d[0], d[0]))
        self.e1=Block(d[0],g,k); self.p1=GridPool(d[0],d[1],0.08)
        self.e2=Block(d[1],g,k); self.p2=GridPool(d[1],d[2],0.16)
        self.e3=Block(d[2],g,k)
        self.u2=nn.Sequential(nn.Linear(d[2]+d[1],d[1]),nn.ReLU()); self.d2=Block(d[1],g,k)
        self.u1=nn.Sequential(nn.Linear(d[1]+d[0],d[0]),nn.ReLU()); self.d1=Block(d[0],g,k)
        self.head=nn.Sequential(nn.LayerNorm(d[0]),nn.Linear(d[0],128),nn.ReLU(),
                                nn.Dropout(0.4),nn.Linear(128,nc))
    def forward(self, x):
        B,C,N = x.shape
        flat  = x.permute(0,2,1).reshape(-1,C).contiguous()
        p0    = flat[:,:3].contiguous()
        b0    = torch.arange(B,device=x.device).repeat_interleave(N)
        h0=self.e1(self.embed(flat),p0,b0)
        h1,p1,b1,c1=self.p1(h0,p0,b0); h1=self.e2(h1,p1,b1)
        h2,p2,b2,c2=self.p2(h1,p1,b1); h2=self.e3(h2,p2,b2)
        u1=self.d2(self.u2(torch.cat([h1,h2[c2]],1)),p1,b1)
        u0=self.d1(self.u1(torch.cat([h0,u1[c1]],1)),p0,b0)
        return self.head(u0).view(B,N,-1).permute(0,2,1)

model = PTv2Seg()
log.info(f"PTv2 parameters: {sum(p.numel() for p in model.parameters()):,}")


# %% [markdown]
# ## Dataset & DataLoader

# %%
from torch.utils.data import Dataset, DataLoader


def _augment_and_pack(feat, lbl, chosen, N, augment):
    """Shared post-processing: crop → chunk-local recenter → augmentation →
    tensor packing. Identical across all 5 variants (unchanged from before)."""
    x = feat[chosen].copy()
    y = lbl[chosen].copy()

    x[:, :3] -= x[:, :3].mean(axis=0, keepdims=True)

    if augment:
        angle = np.random.uniform(0, 2 * np.pi)
        c, sn = np.cos(angle), np.sin(angle)
        R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], np.float32)
        x[:, :3]  = x[:, :3]  @ R.T
        x[:, 4:7] = x[:, 4:7] @ R.T
        if CONFIG.get("aug_jitter", 0) > 0:
            x[:, :3] += np.random.normal(
                0, CONFIG["aug_jitter"], x[:, :3].shape).astype(np.float32)
        if CONFIG.get("aug_scale", 0) > 0:
            s_ = np.float32(1.0 + np.random.uniform(
                -CONFIG["aug_scale"], CONFIG["aug_scale"]))
            x[:, :3] *= s_
        if CONFIG.get("aug_dropout", 0) > 0:
            keep = np.random.rand(N) > CONFIG["aug_dropout"]
            if keep.sum() > N // 2:
                kept = np.flatnonzero(keep)
                fill = np.random.choice(kept, N - len(kept), replace=True)
                order = np.r_[kept, fill]
                x, y = x[order], y[order]

    return torch.from_numpy(x.T).contiguous(), torch.from_numpy(y)


def _knn_sphere_fallback(feat, rng, N, n):
    """Used by every variant when a spatial pool is too sparse."""
    seed = rng.randint(n)
    dist = ((feat[:, :3] - feat[seed, :3]) ** 2).sum(1)
    return np.argpartition(dist, N - 1)[:N]


class PointCloudDataset_RandomGrid(Dataset):
    """V1: baseline — random 6×6 grid cell per __getitem__."""
    def __init__(self, files, augment=True):
        self.files, self.augment = list(files), augment
        self.N, self.chunks = CONFIG["num_points"], CONFIG["chunks_per_cloud"]

    def __len__(self):
        return len(self.files) * self.chunks

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks]
        pts, feat, lbl = get_features(path)
        n = len(feat)
        rng = np.random if self.augment else np.random.RandomState(idx)

        if n <= self.N:
            chosen = rng.choice(n, self.N, replace=True)
        else:
            col = max(np.ptp(feat[:, 0]), np.ptp(feat[:, 1])) / 6.0 + 1e-9
            cx = rng.uniform(feat[:, 0].min(), feat[:, 0].max())
            cy = rng.uniform(feat[:, 1].min(), feat[:, 1].max())
            pool = np.flatnonzero((np.abs(feat[:, 0] - cx) < col) &
                                  (np.abs(feat[:, 1] - cy) < col))
            if len(pool) < self.N // 4:
                chosen = _knn_sphere_fallback(feat, rng, self.N, n)
            else:
                chosen = rng.choice(pool, self.N, replace=len(pool) < self.N)

        return _augment_and_pack(feat, lbl, chosen, self.N, self.augment)


class PointCloudDataset_SequentialGrid(Dataset):
    """V2: deterministic row-major sweep over a 6×6 grid, wraps per file."""
    GRID = 6

    def __init__(self, files, augment=True):
        self.files, self.augment = list(files), augment
        self.N, self.chunks = CONFIG["num_points"], CONFIG["chunks_per_cloud"]

    def __len__(self):
        return len(self.files) * self.chunks

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks]
        cell_idx = idx % self.chunks
        pts, feat, lbl = get_features(path)
        n = len(feat)
        rng = np.random if self.augment else np.random.RandomState(idx)

        if n <= self.N:
            chosen = rng.choice(n, self.N, replace=True)
        else:
            g = self.GRID
            gx, gy = (cell_idx % (g * g)) % g, (cell_idx % (g * g)) // g
            x0, x1 = feat[:, 0].min(), feat[:, 0].max()
            y0, y1 = feat[:, 1].min(), feat[:, 1].max()
            cx = x0 + (gx + 0.5) * (x1 - x0) / g
            cy = y0 + (gy + 0.5) * (y1 - y0) / g
            col = max(x1 - x0, y1 - y0) / g + 1e-9
            pool = np.flatnonzero((np.abs(feat[:, 0] - cx) < col) &
                                  (np.abs(feat[:, 1] - cy) < col))
            if len(pool) < self.N // 4:
                chosen = _knn_sphere_fallback(feat, rng, self.N, n)
            else:
                chosen = rng.choice(pool, self.N, replace=len(pool) < self.N)

        return _augment_and_pack(feat, lbl, chosen, self.N, self.augment)


class PointCloudDataset_SlidingWindow(Dataset):
    """V3: overlapping sliding window, stride = 50% of window size."""
    def __init__(self, files, augment=True, window_frac=1/6, overlap=0.5):
        self.files, self.augment = list(files), augment
        self.N, self.chunks = CONFIG["num_points"], CONFIG["chunks_per_cloud"]
        self.window_frac, self.overlap = window_frac, overlap

    def __len__(self):
        return len(self.files) * self.chunks

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks]
        step_idx = idx % self.chunks
        pts, feat, lbl = get_features(path)
        n = len(feat)
        rng = np.random if self.augment else np.random.RandomState(idx)

        if n <= self.N:
            chosen = rng.choice(n, self.N, replace=True)
        else:
            x0, x1 = feat[:, 0].min(), feat[:, 0].max()
            y0, y1 = feat[:, 1].min(), feat[:, 1].max()
            win = max(x1 - x0, y1 - y0) * self.window_frac + 1e-9
            stride = win * (1 - self.overlap)
            n_steps_x = max(int((x1 - x0) / stride), 1)
            sx, sy = step_idx % n_steps_x, step_idx // n_steps_x
            cx = x0 + win / 2 + sx * stride
            cy = y0 + win / 2 + (sy % max(int((y1 - y0) / stride), 1)) * stride
            pool = np.flatnonzero((np.abs(feat[:, 0] - cx) < win / 2) &
                                  (np.abs(feat[:, 1] - cy) < win / 2))
            if len(pool) < self.N // 4:
                chosen = _knn_sphere_fallback(feat, rng, self.N, n)
            else:
                chosen = rng.choice(pool, self.N, replace=len(pool) < self.N)

        return _augment_and_pack(feat, lbl, chosen, self.N, self.augment)


class PointCloudDataset_CoverageBased(Dataset):
    """V4: least-visited-region-first sampling using a per-file visit count."""
    def __init__(self, files, augment=True):
        self.files, self.augment = list(files), augment
        self.N, self.chunks = CONFIG["num_points"], CONFIG["chunks_per_cloud"]
        self._visits = {}

    def __len__(self):
        return len(self.files) * self.chunks

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks]
        pts, feat, lbl = get_features(path)
        n = len(feat)
        rng = np.random if self.augment else np.random.RandomState(idx)

        if path not in self._visits:
            self._visits[path] = np.zeros(n, np.int32)
        visits = self._visits[path]

        if n <= self.N:
            chosen = rng.choice(n, self.N, replace=True)
        else:
            least_visited = np.flatnonzero(visits == visits.min())
            seed = least_visited[rng.randint(len(least_visited))]
            col = max(np.ptp(feat[:, 0]), np.ptp(feat[:, 1])) / 6.0 + 1e-9
            cx, cy = feat[seed, 0], feat[seed, 1]
            pool = np.flatnonzero((np.abs(feat[:, 0] - cx) < col) &
                                  (np.abs(feat[:, 1] - cy) < col))
            if len(pool) < self.N // 4:
                chosen = _knn_sphere_fallback(feat, rng, self.N, n)
            else:
                chosen = rng.choice(pool, self.N, replace=len(pool) < self.N)

        visits[chosen] += 1
        return _augment_and_pack(feat, lbl, chosen, self.N, self.augment)


class PointCloudDataset_FPSCenters(Dataset):
    """V5: FPS-spread chunk centers + jitter. Best coverage/diversity balance."""
    def __init__(self, files, augment=True):
        self.files, self.augment = list(files), augment
        self.N, self.chunks = CONFIG["num_points"], CONFIG["chunks_per_cloud"]
        self._center_cache = {}

    def __len__(self):
        return len(self.files) * self.chunks

    def __getitem__(self, idx):
        path = self.files[idx // self.chunks]
        pts, feat, lbl = get_features(path)
        n = len(feat)
        rng = np.random if self.augment else np.random.RandomState(idx)

        if n <= self.N:
            chosen = rng.choice(n, self.N, replace=True)
        else:
            if path not in self._center_cache:
                xy = feat[:, :2]
                k = min(self.chunks, n)
                idx0 = np.random.RandomState(0).randint(n)
                d2 = ((xy - xy[idx0]) ** 2).sum(1)
                sel = np.empty(k, np.int64); sel[0] = idx0
                for i in range(1, k):
                    sel[i] = int(d2.argmax())
                    d2 = np.minimum(d2, ((xy - xy[sel[i]]) ** 2).sum(1))
                self._center_cache[path] = xy[sel]

            centers = self._center_cache[path]
            col = max(np.ptp(feat[:, 0]), np.ptp(feat[:, 1])) / 6.0 + 1e-9
            c = centers[rng.randint(len(centers))]
            cx = c[0] + rng.normal(0, col * 0.3)
            cy = c[1] + rng.normal(0, col * 0.3)
            pool = np.flatnonzero((np.abs(feat[:, 0] - cx) < col) &
                                  (np.abs(feat[:, 1] - cy) < col))
            if len(pool) < self.N // 4:
                chosen = _knn_sphere_fallback(feat, rng, self.N, n)
            else:
                chosen = rng.choice(pool, self.N, replace=len(pool) < self.N)

        return _augment_and_pack(feat, lbl, chosen, self.N, self.augment)


PointCloudDataset = PointCloudDataset_SlidingWindow


# %% [markdown]
# ## Training

# %%
CONFIG["class_weights"] = compute_class_weights(TRAIN_FILES)

# %%

SAVE_EVERY = 10
def train_model(model, model_name):
    """Train the model with periodic saving. Resumes automatically if a
    checkpoint exists. Returns the model loaded with the best weights."""

    ckpt_dir  = CONFIG["checkpoint_dir"]
    best_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")

    def latest_periodic():
        """Return (epoch, path) of the most recent periodic checkpoint, or (0, None)."""
        pattern = os.path.join(ckpt_dir, f"{model_name}_epoch_*.pth")
        files   = sorted(glob.glob(pattern))
        if not files:
            return 0, None
        def epoch_of(p):
            try:   return int(os.path.basename(p).split("_epoch_")[1].replace(".pth",""))
            except: return 0
        files.sort(key=epoch_of)
        return epoch_of(files[-1]), files[-1]

    def prune_old_checkpoints(keep=2):
        pattern = os.path.join(ckpt_dir, f"{model_name}_epoch_*.pth")
        files   = sorted(glob.glob(pattern))
        def epoch_of(p):
            try:   return int(os.path.basename(p).split("_epoch_")[1].replace(".pth",""))
            except: return 0
        files.sort(key=epoch_of)
        for old in files[:-keep]:
            try:    os.remove(old); log.info(f"Removed old checkpoint: {old}")
            except: pass

    _nw = CONFIG.get("num_workers", 4)
    train_loader = DataLoader(
        PointCloudDataset(TRAIN_FILES, augment=True),
        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True,
        num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0))
    val_loader = DataLoader(
        PointCloudDataset(VAL_FILES, augment=False),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0))

    model.to(DEVICE)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=CONFIG.get("sched_factor", 0.5),
        patience=CONFIG.get("sched_patience", 10),
        min_lr=CONFIG.get("min_lr", 1e-6))
    use_amp = CONFIG.get("use_amp", True) and DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if CONFIG.get("class_weights") is not None:
        cw = torch.tensor(CONFIG["class_weights"], dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        log.info(f"Using class weights: {CONFIG['class_weights']}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    start_epoch = 1
    best_miou   = -1.0
    no_improve  = 0
    history     = []

    resume_epoch, resume_path = latest_periodic()
    if resume_path:
        log.info(f"Resuming from {resume_path} (epoch {resume_epoch})")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou",   -1.0)
        no_improve  = ckpt.get("no_improve",   0)
        history     = ckpt.get("history",      [])
        log.info(f"Resumed: start_epoch={start_epoch}  best_miou={best_miou:.4f}")
    else:
        log.info(f"No checkpoint found — training from scratch.")

    if start_epoch > CONFIG["epochs"]:
        log.info("Training already complete. Loading best model.")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        return model

    run_name = f"{model_name}_seg"
    with (mlflow.start_run(run_name=run_name) if MLFLOW_OK
          else open(os.devnull, "w")) as _:

        if MLFLOW_OK:
            mlflow.log_params({
                "model"       : model_name,
                "epochs"      : CONFIG["epochs"],
                "batch_size"  : CONFIG["batch_size"],
                "num_points"  : CONFIG["num_points"],
                "lr"          : CONFIG["lr"],
                "start_epoch" : start_epoch,
            })

        accumulation_steps = CONFIG.get("grad_accum", 4)

        for epoch in range(start_epoch, CONFIG["epochs"] + 1):

            model.train()
            optimizer.zero_grad()
            
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out  = model(x)
                    loss = loss_fn(out, y)

                train_loss_sum += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += y.numel()

                scaler.scale(loss / accumulation_steps).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            epoch_train_loss = train_loss_sum / len(train_loader.dataset)
            epoch_train_acc = train_correct / train_total

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            all_true, all_pred = [], []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out  = model(x)
                        loss = loss_fn(out, y)
                    
                    val_loss_sum += loss.item() * x.size(0)
                    preds = out.argmax(dim=1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.numel()

                    all_pred.extend(preds.cpu().numpy().ravel())
                    all_true.extend(y.cpu().numpy().ravel())

            epoch_val_loss = val_loss_sum / len(val_loader.dataset)
            epoch_val_acc = val_correct / val_total
            
            miou = compute_miou(np.array(all_true), np.array(all_pred), NUM_CLASSES)
            dice = compute_dice(np.array(all_true), np.array(all_pred), NUM_CLASSES)
            
            history.append({"epoch": epoch, "val_miou": miou, "val_loss": epoch_val_loss})
            
            log.info(
                f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
                f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | "
                f"mIoU: {miou:.4f} | Dice: {dice:.4f}"
            )

            if MLFLOW_OK:
                mlflow.log_metrics({
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "train_acc": epoch_train_acc,
                    "val_acc": epoch_val_acc,
                    "val_miou": miou,
                    "val_dice": dice
                }, step=epoch)

            scheduler.step(miou)
            if miou > best_miou + 1e-4:
                best_miou  = miou
                no_improve = 0
                torch.save({"model_state": model.state_dict(),
                            "best_val_miou": best_miou,
                            "in_channels": CONFIG["in_channels"],
                            "num_classes": NUM_CLASSES,
                            "num_points": CONFIG["num_points"],
                            "use_linearity" : CONFIG["use_linearity"],
                            "use_planarity" : CONFIG["use_planarity"],
                            "use_sphericity": CONFIG["use_sphericity"],
                            "use_rgb"       : CONFIG["use_rgb"]}, best_path)
                log.info(f"  ✓ New best mIoU {best_miou:.4f} → saved {best_path}")
            else:
                no_improve += 1


            if epoch % SAVE_EVERY == 0 or epoch == CONFIG["epochs"]:
                periodic_path = os.path.join(ckpt_dir,
                                             f"{model_name}_epoch_{epoch:04d}.pth")
                torch.save({
                    "model_state"     : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "epoch"           : epoch,
                    "best_miou"       : best_miou,
                    "no_improve"      : no_improve,
                    "history"         : history,
                    "in_channels"     : CONFIG["in_channels"],
                    "num_classes"     : NUM_CLASSES,
                    "num_points"      : CONFIG["num_points"],
                    "use_linearity"   : CONFIG["use_linearity"],
                    "use_planarity"   : CONFIG["use_planarity"],
                    "use_sphericity"  : CONFIG["use_sphericity"],
                    "use_rgb"         : CONFIG["use_rgb"],
                }, periodic_path)
                log.info(f"  💾 Periodic checkpoint saved → {periodic_path}")
                prune_old_checkpoints(keep=2)

            if no_improve >= CONFIG["patience"]:
                log.info(f"Early stop at epoch {epoch} "
                         f"(no improvement for {CONFIG['patience']} epochs)")
                break

        if MLFLOW_OK:
            mlflow.log_metric("best_val_miou", best_miou)

    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        log.info(f"Loaded best model (mIoU={best_miou:.4f}) from {best_path}")
    model.to("cpu")
    return model


# %%
model = train_model(model, MODEL_NAME)

# %% [markdown]
# ## Visualization helper

# %%
def visualize_segmentation(points, predictions, title="Segmentation"):
    """Open an Open3D window showing the segmentation result."""
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[predictions == CONFIG["target_class"]] = [0.0, 0.8, 0.0]
    colors[predictions != CONFIG["target_class"]] = [0.8, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    center = points.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector(
        (points - center).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    span = float((points.max(0) - points.min(0)).max()) * 0.5
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=span)

    n_target = int((predictions == CONFIG["target_class"]).sum())
    n_other  = len(predictions) - n_target
    full_title = (f"{title}  |  green(target)={n_target:,}  "
                  f"red(others)={n_other:,}")
    o3d.visualization.draw_geometries([pcd, axes],
                                       window_name=full_title,
                                       width=1280, height=800)


# %% [markdown]
# ## Inference helper

# %%
from scipy.spatial import cKDTree as _InferKDTree

@torch.no_grad()
def predict_full_cloud(model, path, overlap_factor=2):
    """Label every point using kNN-sphere chunks (same spatial nature as the
    Dataset's chunks — FIX 1). Random-permutation chunking mixes points from
    anywhere in the cloud into one chunk, which is NOT what the model saw
    during training and degrades predictions, especially at boundaries.

    overlap_factor controls how many times each point is (on average)
    covered by different chunks — >1 gives overlapping coverage so a
    majority vote can smooth out boundary artifacts (FIX 3/4)."""
    model.eval()
    pts, feat, lbl = get_features(path)
    n = len(feat)
    N = CONFIG["num_points"]
    B = CONFIG["batch_size"]

    model.to(DEVICE)

    if n <= N:
        pad_idx = np.random.RandomState(0).choice(n, N, replace=True)
        x   = torch.from_numpy(feat[pad_idx][None].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy().ravel()
        preds = np.zeros(n, np.int64)
        preds[pad_idx] = out
        model.to("cpu")
        return pts, preds, lbl

    tree      = _InferKDTree(feat[:, :3])
    covered   = np.zeros(n, bool)
    vote_sum  = np.zeros((n, NUM_CLASSES), np.int32)
    rng       = np.random.RandomState(0)
    batch_ids = []

    def flush():
        if not batch_ids:
            return
        x   = torch.from_numpy(
                  feat[np.stack(batch_ids)].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy()
        for ids, o in zip(batch_ids, out):
            np.add.at(vote_sum, (ids, o), 1)
        batch_ids.clear()

    target_visits = max(1, overlap_factor)
    visit_count = np.zeros(n, np.int32)
    max_iters = int(np.ceil(n / N * target_visits)) + 4
    for _ in range(max_iters):
        if (visit_count >= target_visits).all():
            break
        least = np.flatnonzero(visit_count == visit_count.min())
        seed  = least[rng.randint(len(least))]
        _, idx = tree.query(pts[seed], k=N)
        idx = np.atleast_1d(idx)
        batch_ids.append(idx)
        visit_count[idx] += 1
        covered[idx] = True
        if len(batch_ids) == B:
            flush()
    flush()

    if not covered.all() and covered.any():
        missing     = np.flatnonzero(~covered)
        covered_idx = np.flatnonzero(covered)
        _, nn_local = _InferKDTree(pts[covered_idx]).query(pts[missing], k=1)
        vote_sum[missing] = vote_sum[covered_idx[nn_local]]

    preds = vote_sum.argmax(axis=1).astype(np.int64)
    model.to("cpu")
    return pts, preds, lbl


# %% [markdown]
# ## Per-file Train/Val Report (log only, no saving)

# %%
log.info("=== TRAIN/VAL PER-FILE REPORT ===")

trval_names, trval_true, trval_pred = [], [], []
for path in TRAIN_FILES:
    name = os.path.splitext(os.path.basename(path))[0]
    pts, preds, lbl = predict_full_cloud(model, path)
    trval_names.append(f"{name} [train]"); trval_true.append(lbl); trval_pred.append(preds)

per_file_report(trval_names, trval_true, trval_pred, tag="TRAIN",
                target_class=CONFIG["target_class"])

val_names, val_true, val_pred = [], [], []
for path in VAL_FILES:
    name = os.path.splitext(os.path.basename(path))[0]
    pts, preds, lbl = predict_full_cloud(model, path)
    val_names.append(f"{name} [val]"); val_true.append(lbl); val_pred.append(preds)

per_file_report(val_names, val_true, val_pred, tag="VAL",
                target_class=CONFIG["target_class"])

# %% [markdown]
# ## Volume Estimation (TIN)
# 
# Segmentation-এর পরে target points থেকে volume বের করা হয়।
# 
# **Pipeline (সব `vol_TIN()`-এর ভেতরে):**
# 1. **SOR** — noisy বিচ্ছিন্ন point বাদ (statistical distance filter)
# 2. **DBSCAN largest cluster** — stray mislabeled points বাদ
# 3. **TIN integration** — 2D Delaunay → Σ Area₂D(triangle) × mean height
#    - Floor baseline = scanned z-এর `floor_pct` percentile (outer-shell scan-এ আলাদা floor point থাকে না)
#    - কোনো alpha clipping নেই — DBSCAN-ই stray point সামলায়; clipping করলে কিনারার বৈধ triangle কেটে গিয়ে volume কমে যেত
# 4. **Bootstrap CI** — ৮০ বার subsample করে std + 95% confidence interval
# 

# %%
from scipy.spatial import Delaunay, cKDTree


def sor_filter(pts, k=16, std_ratio=2.0):
    """Statistical Outlier Removal: mean-kNN-distance > mean + std_ratio*std
    হলে সেই point বাদ। Returns (filtered_pts, kept_mask)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) <= k + 1:
        return pts, np.ones(len(pts), bool)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    mean_dist = d[:, 1:].mean(axis=1)
    thr  = mean_dist.mean() + std_ratio * mean_dist.std()
    mask = mean_dist <= thr
    return pts[mask], mask


def extract_main_cluster(pts, eps=0.15, min_samples=8):
    """DBSCAN দিয়ে সবচেয়ে বড় connected cluster রাখা — stray points বাদ।
    এর পরে আর alpha clipping লাগে না।"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < min_samples * 2:
        return pts
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
        valid = labels[labels >= 0]
        if valid.size == 0:
            return pts
        main_lbl = np.bincount(valid).argmax()
        result = pts[labels == main_lbl]
        return result if len(result) >= min_samples else pts
    except Exception as e:
        log.warning(f"[TIN] cluster extraction failed ({e}) -> using all points")
        return pts


def tin_integrate(pts, floor_pct=0.5):
    """Delaunay TIN integration.
    floor baseline = scanned z-এর floor_pct percentile (pile-এর base প্রান্ত)।
    Volume = Σ Area2D(triangle) × mean_height(triangle)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 4:
        return float("nan")

    floor_z = np.percentile(pts[:, 2], floor_pct)
    h  = pts[:, 2] - floor_z
    xy = pts[:, :2]
    try:
        tri = Delaunay(xy)
    except Exception as e:
        log.warning(f"[TIN] Delaunay failed ({e})")
        return float("nan")

    t = tri.simplices
    p0, p1, p2 = xy[t[:, 0]], xy[t[:, 1]], xy[t[:, 2]]
    h0, h1, h2 = h[t[:, 0]], h[t[:, 1]], h[t[:, 2]]

    cross  = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
             - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    area2d = np.abs(cross) / 2.0
    mean_h = (h0 + h1 + h2) / 3.0
    valid  = mean_h > 0
    return float(np.sum(area2d[valid] * mean_h[valid]))


def bootstrap_tin_confidence(pts, n_bootstrap=80, subsample_frac=0.85,
                             floor_pct=0.5, seed=0):
    """Bootstrap: বারবার subsample করে TIN চালিয়ে uncertainty মাপা।
    Returns {'std', 'ci_low', 'ci_high'} বা None।"""
    rng = np.random.RandomState(seed)
    n = len(pts)
    n_sub = max(10, int(n * subsample_frac))
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n_sub, replace=False)
        v = tin_integrate(pts[idx], floor_pct=floor_pct)
        if np.isfinite(v) and v > 0:
            estimates.append(v)
    if len(estimates) < 5:
        return None
    arr = np.array(estimates)
    return {"std":     round(float(arr.std()), 5),
            "ci_low":  round(float(np.percentile(arr, 2.5)), 5),
            "ci_high": round(float(np.percentile(arr, 97.5)), 5)}


def vol_TIN(pts, sor_k=16, sor_std=2.0, cluster_eps=0.15,
            floor_pct=0.5, run_bootstrap=True, n_bootstrap=80):
    """সম্পূর্ণ volume pipeline: SOR → largest cluster → TIN [→ bootstrap CI].
    Returns (volume, confidence_dict_or_None). একক = pts-এর এককের ঘন (m³)।"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 10:
        return float("nan"), None

    pts, _ = sor_filter(pts, k=sor_k, std_ratio=sor_std)
    if len(pts) < 10:
        return float("nan"), None

    pts = extract_main_cluster(pts, eps=cluster_eps)
    if len(pts) < 10:
        return float("nan"), None

    volume = tin_integrate(pts, floor_pct=floor_pct)

    conf = None
    if run_bootstrap and len(pts) >= 20:
        conf = bootstrap_tin_confidence(pts, n_bootstrap=n_bootstrap,
                                        floor_pct=floor_pct)
    return float(volume), conf


# %% [markdown]
# ## Testing  (held-out test files with labels)

# %%
log.info("=== TESTING ===")
test_names, test_true, test_pred = [], [], []
for path in TEST_FILES:
    name = os.path.splitext(os.path.basename(path))[0]
    pts, preds, lbl = predict_full_cloud(model, path)
    test_names.append(name); test_true.append(lbl); test_pred.append(preds)

per_file_report(test_names, test_true, test_pred, tag="TEST",
                target_class=CONFIG["target_class"])

for path in TEST_FILES[:CONFIG["max_vis_files"]]:
    name = os.path.splitext(os.path.basename(path))[0]
    pts, preds, _ = predict_full_cloud(model, path)

    target_pts   = pts[preds == CONFIG["target_class"]]
    volume, conf = vol_TIN(target_pts)

    n_tot = len(pts)
    n_tgt = len(target_pts)
    print(f"\n{'─'*55}")
    print(f"  File          : {name}")
    print(f"  Total Points  : {n_tot:,}")
    print(f"  Target Points : {n_tgt:,}")
    print(f"  Other Points  : {n_tot - n_tgt:,}")
    print(f"  Target Ratio  : {n_tgt / max(n_tot, 1) * 100:.2f}%")
    print(f"  Est. Volume   : {volume:.4f} m³")
    if conf:
        print(f"  95% CI        : [{conf['ci_low']:.4f} – {conf['ci_high']:.4f}] m³"
              f"  (±{conf['std']:.4f})")
    print(f"{'─'*55}")

    visualize_segmentation(pts, preds,
                           title=f"{MODEL_NAME} | {name} | V={volume:.4f} m³")


# %% [markdown]
# ## Final Inference  (`data/test`, no labels)

# %%
if INFER_FILES:
    log.info("=== FINAL INFERENCE ===")
    infer_names, infer_true, infer_pred, infer_pts = [], [], [], []
    for path in INFER_FILES:
        name = os.path.splitext(os.path.basename(path))[0]
        pts, preds, lbl = predict_full_cloud(model, path)
        infer_names.append(name)
        infer_true.append(lbl if lbl is not None and lbl.any() else None)
        infer_pred.append(preds)
        infer_pts.append(pts)

    per_file_report(infer_names, infer_true, infer_pred, tag="INFERENCE",
                    target_class=CONFIG["target_class"])

    for name, pts, preds in zip(infer_names, infer_pts, infer_pred):
        target_pts   = pts[preds == CONFIG["target_class"]]
        volume, conf = vol_TIN(target_pts)

        n_tot = len(pts)
        n_tgt = len(target_pts)
        print(f"\n{'─'*55}")
        print(f"  File          : {name}")
        print(f"  Total Points  : {n_tot:,}")
        print(f"  Target Points : {n_tgt:,}")
        print(f"  Other Points  : {n_tot - n_tgt:,}")
        print(f"  Target Ratio  : {n_tgt / max(n_tot, 1) * 100:.2f}%")
        print(f"  Est. Volume   : {volume:.4f} m³")
        if conf:
            print(f"  95% CI        : [{conf['ci_low']:.4f} – {conf['ci_high']:.4f}] m³"
                  f"  (±{conf['std']:.4f})")
        print(f"{'─'*55}")

        visualize_segmentation(pts, preds,
                               title=f"INFERENCE | {MODEL_NAME} | {name} | "
                                     f"V={volume:.4f} m³")
else:
    log.info("data/test is empty — skipping inference.")


# %%
import torch
torch.version.cuda

# %%



