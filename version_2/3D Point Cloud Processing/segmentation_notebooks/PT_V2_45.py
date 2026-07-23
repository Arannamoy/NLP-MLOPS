# %%
import os, io, glob, platform, glob, copy, random, logging
import requests
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
import io, os, glob, socket, urllib.parse
import gdown
USE_GOOGLE_DRIVE = False
GDRIVE_FOLDER_ID = "https://drive.google.com/drive/folders/1fUCLmHNBI5gCP4w1Z-qA49Lkq4V_8xhC?usp=sharing"
LOCAL_DATA_DIR   = "../data/train_v1"
RAM_STORE = {}
def _load_from_drive(folder_id):
    folder_id = urllib.parse.urlparse(folder_id).path.rstrip("/").split("/")[-1]
    _old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(60)
    try:
        try:
            entries = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
        except (TimeoutError, socket.timeout) as e:
            raise RuntimeError(
                f"Timed out listing Drive folder {folder_id!r} after 60s. "
                f"Check your network connection, or that the folder is "
                f"actually shared as 'Anyone with the link'.") from e
    finally:
        socket.setdefaulttimeout(_old_timeout)
    entries = [e for e in entries
              if e.local_path.lower().endswith((".las", ".laz"))]
    if not entries:
        raise RuntimeError(
            "No .las/.laz files found in that folder. Check GDRIVE_FOLDER_ID "
            "and that the folder is shared as 'Anyone with the link'.")
    for e in entries:
        name = os.path.basename(e.local_path)
        buf = io.BytesIO()
        _old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(60)
        try:
            try:
                gdown.download(id=e.id, output=buf, quiet=True)
            except (TimeoutError, socket.timeout) as ex:
                raise RuntimeError(
                    f"Timed out downloading {name!r} after 60s — Drive may be "
                    f"rate-limiting or the connection stalled. Try again, or "
                    f"download this one file manually to confirm connectivity."
                ) from ex
        finally:
            socket.setdefaulttimeout(_old_timeout)
        buf.seek(0)
        RAM_STORE[name] = buf.getvalue()
        log.info(f"  downloaded {name} "
                 f"({len(RAM_STORE[name]) / (1024**2):.1f} MB) into RAM")
def _list_local_files(folder):
    files = []
    for ext in (".las", ".laz"):
        files += glob.glob(os.path.join(folder, "*" + ext))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No .las/.laz files found in {folder!r}.")
    log.info(f"Found {len(files)} local file(s) in {folder!r} "
             f"(read directly from disk on demand — not duplicated into RAM)")
    return files
def load_dataset_source():
    if USE_GOOGLE_DRIVE:
        log.info(f"Loading dataset from Google Drive (folder {GDRIVE_FOLDER_ID!r})…")
        _load_from_drive(GDRIVE_FOLDER_ID)
        files = sorted(RAM_STORE.keys())
        total_mb = sum(len(b) for b in RAM_STORE.values()) / (1024**2)
        log.info(f"Loaded {len(files)} files into RAM_STORE ({total_mb:.1f} MB total)")
    else:
        files = _list_local_files(LOCAL_DATA_DIR)
    return files
ALL_FILES = load_dataset_source()

# %%
CONFIG = {
    "checkpoint_dir" : "../checkpoints/PT_v2_12_channel_v45",
    "num_classes"    : 2,
    "target_class"   : 1,
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "cv_folds"       : 5,
    "seed"           : 42,
    "epochs"         : 1500,
    "patience"       : 100,
    "batch_size"     : 4,
    "grad_accum"     : 1,
    "num_points"     : 16384,
    "chunks_per_cloud": 100,
    "lr"             : 2e-3,
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
    "use_linearity"    : True,
    "use_planarity"    : True,
    "use_sphericity"   : True,
    "use_verticality"  : True,
    "use_eigenentropy" : True,
    "geom_radius"      : 0.06,
    "use_rgb"          : False,
    "in_channels"      : 7,
    "num_workers"    : (0 if platform.system() == "Windows" else 4),
    "weight_decay"        : 1e-4,
    "auto_regularize"     : True,
    "auto_reg_patience"   : 15,
    "auto_reg_wd_factor"  : 2.0,
    "auto_reg_wd_max"     : 1e-2,
    "auto_reg_dropout_step": 0.05,
    "auto_reg_dropout_max": 0.6,
    "auto_reg_max_triggers": 3,
    "use_amp"        : True,
    "class_weights"  : None,
    "max_vis_files"  : 3,
    "mlflow_uri"     : "http://localhost:5000",
    "mlflow_experiment": "PT_v2_12_channel_v45",
}
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
CONFIG["in_channels"] = (7 + (1 if CONFIG["use_linearity"] else 0)
                           + (1 if CONFIG["use_planarity"] else 0)
                           + (1 if CONFIG["use_sphericity"] else 0)
                           + (1 if CONFIG["use_verticality"] else 0)
                           + (1 if CONFIG["use_eigenentropy"] else 0)
                           + (3 if CONFIG["use_rgb"] else 0))
NUM_CLASSES = CONFIG["num_classes"]
def compute_class_weights(files, cap=10.0):
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
    if path in RAM_STORE:
        las = laspy.read(io.BytesIO(RAM_STORE[path]))
    else:
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
from sklearn.model_selection import KFold
all_files = sorted(ALL_FILES)
assert all_files, "ALL_FILES is empty — did load_dataset_source() run successfully?"
rng   = np.random.RandomState(CONFIG["seed"])
order = rng.permutation(len(all_files))
n_test = max(1, int(len(all_files) * CONFIG["test_ratio"]))
TEST_FILES = [all_files[i] for i in order[:n_test]]
remaining  = [all_files[i] for i in order[n_test:]]
if CONFIG.get("cv_folds", 1) <= 1:
    n_val = max(1, int(len(all_files) * CONFIG["val_ratio"]))
    VAL_FILES   = remaining[:n_val]
    TRAIN_FILES = remaining[n_val:]
    CV_FOLDS = None
    log.info(f"Single split: train={len(TRAIN_FILES)}  val={len(VAL_FILES)}  "
             f"test={len(TEST_FILES)}")
else:
    kf = KFold(n_splits=CONFIG["cv_folds"], shuffle=True,
              random_state=CONFIG["seed"])
    CV_FOLDS = []
    for tr_idx, va_idx in kf.split(remaining):
        CV_FOLDS.append(([remaining[i] for i in tr_idx],
                         [remaining[i] for i in va_idx]))
    TRAIN_FILES, VAL_FILES = CV_FOLDS[0]
    log.info(f"{CONFIG['cv_folds']}-fold CV on {len(remaining)} files "
             f"(test={len(TEST_FILES)} held out, never in any fold)")
    for i, (tf, vf) in enumerate(CV_FOLDS):
        log.info(f"  fold {i}: train={len(tf)}  val={len(vf)}")
INFER_FILES = []
log.info(f"train={len(TRAIN_FILES)}  val={len(VAL_FILES)}  "
         f"test={len(TEST_FILES)}  inference={len(INFER_FILES)}")

# %%
USE_GOOGLE_DRIVE_INFER = False
GDRIVE_INFER_FOLDER_ID = "https://drive.google.com/drive/folders/1XBszFp5F7ZP-ucTAf-Pp8hdQhPHHuMn1?usp=sharing"
LOCAL_INFER_DIR        = "../data/test"
def load_inference_files():
    try:
        if USE_GOOGLE_DRIVE_INFER:
            log.info(f"Loading inference set from Google Drive "
                     f"(folder {GDRIVE_INFER_FOLDER_ID!r})…")
            before = set(RAM_STORE.keys())
            _load_from_drive(GDRIVE_INFER_FOLDER_ID)
            new_keys = sorted(set(RAM_STORE.keys()) - before)
            log.info(f"Loaded {len(new_keys)} inference file(s) into RAM_STORE "
                     f"(kept separate from train/val/test)")
            return new_keys
        else:
            log.info(f"Loading inference set from local disk "
                     f"({LOCAL_INFER_DIR!r})…")
            return _list_local_files(LOCAL_INFER_DIR)
    except RuntimeError as e:
        log.warning(f"No inference set loaded ({e}) — INFER_FILES will be empty.")
        return []
INFER_FILES = load_inference_files()

# %%
def _geom_features(points, radius=0.06, min_neighbors=8,
                   want_linear=True, want_planar=False, want_sphere=False,
                   want_vert=False, want_entropy=False):
    from scipy.spatial import cKDTree as _GeomTree
    n = len(points)
    lin = np.zeros(n, np.float32) if want_linear else None
    pla = np.zeros(n, np.float32) if want_planar else None
    sph = np.zeros(n, np.float32) if want_sphere else None
    vert = np.zeros(n, np.float32) if want_vert else None
    ent  = np.zeros(n, np.float32) if want_entropy else None
    if not (want_linear or want_planar or want_sphere or want_vert or want_entropy):
        return lin, pla, sph, vert, ent
    need_evecs = want_vert
    tree = _GeomTree(points)
    lists = tree.query_ball_point(points, r=radius, workers=-1)
    EIG_FLOOR = 1e-6 # prevent ratio-explosion
    for i, nbrs in enumerate(lists):
        if len(nbrs) < min_neighbors:
            continue
        nb = points[nbrs]
        c = nb - nb.mean(0)
        cov = (c.T @ c) / len(nbrs)
        if need_evecs:
            evals, evecs = np.linalg.eigh(cov)
            l3, l2, l1 = evals
        else:
            l3, l2, l1 = np.linalg.eigvalsh(cov)
        if l1 < EIG_FLOOR:
            continue
        l1c = l1
        # l1c = max(l1, 1e-9)
        if want_linear:  lin[i] = (l1 - l2) / l1c
        if want_planar:  pla[i] = (l2 - l3) / l1c
        if want_sphere:  sph[i] = l3 / l1c
        if want_vert:
            normal_proxy = evecs[:, 0]
            vert[i] = 1.0 - abs(normal_proxy[2])
        if want_entropy:
            s = l1 + l2 + l3
            if s > 1e-9:
                p = np.clip(np.array([l1, l2, l3]) / s, 1e-12, None)
                ent[i] = -(p * np.log(p)).sum()
    return lin, pla, sph, vert, ent
def make_features(points, rgb=None):
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
    want_vert = CONFIG.get("use_verticality", False)
    want_ent  = CONFIG.get("use_eigenentropy", False)
    if want_lin or want_pla or want_sph or want_vert or want_ent:
        lin, pla, sph, vert, ent = _geom_features(
            points, radius=CONFIG.get("geom_radius", 0.06),
            want_linear=want_lin, want_planar=want_pla, want_sphere=want_sph,
            want_vert=want_vert, want_entropy=want_ent)
        if want_lin:  cols.append(lin[:, None])
        if want_pla:  cols.append(pla[:, None])
        if want_sph:  cols.append(sph[:, None])
        if want_vert: cols.append(vert[:, None])
        if want_ent:  cols.append(ent[:, None])
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
    vox = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((cnt.size, 3), np.float64); np.add.at(sums, inv, pts)
    d2 = ((pts - (sums / cnt[:, None])[inv]) ** 2).sum(1)
    order = np.lexsort((d2, inv))
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return np.sort(order[first])
def _sor_keep(pts, k=16, std_ratio=2.0):
    from scipy.spatial import cKDTree as _SORTree
    n = len(pts)
    if n <= k + 1:
        return np.ones(n, bool)
    d, _ = _SORTree(pts).query(pts, k=k + 1)
    mean_dist = d[:, 1:].mean(axis=1)
    thr = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= thr
def _cluster_keep(pts, eps, min_samples=8, min_cluster_size=100):
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
            bool(CONFIG.get("use_verticality", False)),
            bool(CONFIG.get("use_eigenentropy", False)),
            round(float(CONFIG.get("geom_radius", 0.06)), 4),
            bool(CONFIG.get("use_rgb", False)))
def get_features(path):
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
    ious = []
    for c in range(num_classes):
        tp = int(((true_labels == c) & (pred_labels == c)).sum())
        fp = int(((true_labels != c) & (pred_labels == c)).sum())
        fn = int(((true_labels == c) & (pred_labels != c)).sum())
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else 0.0
def compute_dice(true_labels, pred_labels, num_classes):
    dices = []
    for c in range(num_classes):
        tp = int(((true_labels == c) & (pred_labels == c)).sum())
        fp = int(((true_labels != c) & (pred_labels == c)).sum())
        fn = int(((true_labels == c) & (pred_labels != c)).sum())
        if tp + fp + fn > 0:
            dices.append((2 * tp) / (2 * tp + fp + fn))
    return float(np.mean(dices)) if dices else 0.0
def per_file_report(names, true_list, pred_list, tag="", target_class=1):
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
def build_model():
    return PTv2Seg()
model = build_model()
log.info(f"PTv2 parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Dataset & DataLoader

# %%
from torch.utils.data import Dataset, DataLoader
def _augment_and_pack(feat, lbl, chosen, N, augment):
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
    seed = rng.randint(n)
    dist = ((feat[:, :3] - feat[seed, :3]) ** 2).sum(1)
    return np.argpartition(dist, N - 1)[:N]
class PointCloudDataset_RandomGrid(Dataset):
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
import time
SAVE_EVERY = 10
LAST_RUN_ID = None
def train_model(model, model_name):
    ckpt_dir  = CONFIG["checkpoint_dir"]
    best_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
    def latest_periodic():
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
                                  weight_decay=CONFIG.get("weight_decay", 1e-4))
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
    auto_reg_no_improve = 0
    auto_reg_triggers   = 0
    resume_epoch, resume_path = latest_periodic()
    if resume_path:
        log.info(f"Resuming from {resume_path} (epoch {resume_epoch})")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        else:
            log.warning("Checkpoint has no scheduler_state (older checkpoint?) — "
                       "the LR-plateau counter is starting fresh from this resume, "
                       "so LR may not reduce as promptly as it should have.")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou",   -1.0)
        no_improve  = ckpt.get("no_improve",   0)
        history     = ckpt.get("history",      [])
        auto_reg_no_improve = ckpt.get("auto_reg_no_improve", 0)
        auto_reg_triggers   = ckpt.get("auto_reg_triggers", 0)
        if auto_reg_triggers > 0:
            wd = min(CONFIG.get("weight_decay", 1e-4) *
                     (CONFIG.get("auto_reg_wd_factor", 2.0) ** auto_reg_triggers),
                     CONFIG.get("auto_reg_wd_max", 1e-2))
            for g in optimizer.param_groups:
                g["weight_decay"] = wd
            dp_add = CONFIG.get("auto_reg_dropout_step", 0.05) * auto_reg_triggers
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = min(m.p + dp_add, CONFIG.get("auto_reg_dropout_max", 0.6))
            log.info(f"Restored auto-regularization state: {auto_reg_triggers} "
                     f"prior trigger(s), weight_decay={wd:.5f}")
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
            global LAST_RUN_ID
            LAST_RUN_ID = mlflow.active_run().info.run_id

            import subprocess as _sp
            try:
                _git_hash = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                             stderr=_sp.DEVNULL).decode().strip()
            except Exception:
                _git_hash = "unknown"
            try:
                _git_branch = _sp.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                               stderr=_sp.DEVNULL).decode().strip()
            except Exception:
                _git_branch = "unknown"
            _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            _dropout_p = None
            for _m in model.modules():
                if isinstance(_m, nn.Dropout):
                    _dropout_p = _m.p
                    break

            mlflow.set_tags({
                "architecture"     : model_name,
                "git_branch"       : _git_branch,
                "dataset_version"  : CONFIG.get("dataset_version", "v1"),
            })

            mlflow.log_params({
                "model"           : model_name,
                "dataset"         : os.path.basename(os.path.normpath(
                                     CONFIG.get("local_data_dir", "unknown_dataset")))
                                     if "local_data_dir" in CONFIG else "wood_powder_lidar",
                "n_train_files"   : len(TRAIN_FILES),
                "n_val_files"     : len(VAL_FILES),
                "n_test_files"    : len(TEST_FILES),
                "in_channels"     : CONFIG["in_channels"],
                "num_classes"     : NUM_CLASSES,
                "epochs"          : CONFIG["epochs"],
                "patience"        : CONFIG["patience"],
                "batch_size"      : CONFIG["batch_size"],
                "grad_accum"      : CONFIG["grad_accum"],
                "num_points"      : CONFIG["num_points"],
                "lr"              : CONFIG["lr"],
                "cv_folds"        : CONFIG.get("cv_folds", 1),
                "auto_regularize" : CONFIG.get("auto_regularize", False),
                "start_epoch"     : start_epoch,
                "optimizer_type"  : "AdamW",
                "weight_decay"    : CONFIG.get("weight_decay", 1e-4),
                "dropout"         : _dropout_p,
                "seed"            : CONFIG.get("seed", 42),
                "gpu_name"        : _gpu_name,
                "git_commit"      : _git_hash,
                "voxel_size"      : CONFIG.get("voxel_size", 0.01),
                "sor_k"           : CONFIG.get("sor_k", 16),
                "sor_std"         : CONFIG.get("sor_std", 2.0),
                "cluster_eps"         : CONFIG.get("cluster_eps", 0.03),
                "cluster_min_size"    : CONFIG.get("cluster_min_size", 100),
                "geom_radius"         : CONFIG.get("geom_radius", 0.06),
                "use_linearity"       : CONFIG.get("use_linearity", False),
                "use_planarity"       : CONFIG.get("use_planarity", False),
                "use_sphericity"      : CONFIG.get("use_sphericity", False),
                "use_verticality"     : CONFIG.get("use_verticality", False),
                "use_eigenentropy"    : CONFIG.get("use_eigenentropy", False),
                "use_rgb"             : CONFIG.get("use_rgb", False),
            })
        accumulation_steps = CONFIG.get("grad_accum", 4)
        for epoch in range(start_epoch, CONFIG["epochs"] + 1):
            _epoch_t0 = time.time()
            model.train()
            optimizer.zero_grad()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            train_conf = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in range(NUM_CLASSES)}
            _grad_norms = []
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out  = model(x)
                    loss = loss_fn(out, y)
                if not torch.isfinite(loss):
                    log.warning(f"  epoch {epoch} batch {i}: non-finite loss, skipping")
                    optimizer.zero_grad()
                    continue
                train_loss_sum += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += y.numel()
                for c in range(NUM_CLASSES):
                    train_conf[c]["tp"] += int(((preds == c) & (y == c)).sum().item())
                    train_conf[c]["fp"] += int(((preds == c) & (y != c)).sum().item())
                    train_conf[c]["fn"] += int(((preds != c) & (y == c)).sum().item())
                    train_conf[c]["tn"] += int(((preds != c) & (y != c)).sum().item())
                scaler.scale(loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    _gn = 0.0
                    for _p in model.parameters():
                        if _p.grad is not None:
                            _gn += _p.grad.data.norm(2).item() ** 2
                    _grad_norms.append(_gn ** 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_train_loss = train_loss_sum / len(train_loader.dataset)
            epoch_train_acc = train_correct / train_total

            _train_ious, _train_dices_ = [], []
            for c in range(NUM_CLASSES):
                _tp, _fp, _fn = train_conf[c]["tp"], train_conf[c]["fp"], train_conf[c]["fn"]
                if _tp + _fp + _fn > 0:
                    _train_ious.append(_tp / (_tp + _fp + _fn))
                if 2 * _tp + _fp + _fn > 0:
                    _train_dices_.append(2 * _tp / (2 * _tp + _fp + _fn))
            train_miou = float(np.mean(_train_ious)) if _train_ious else 0.0
            train_dice = float(np.mean(_train_dices_)) if _train_dices_ else 0.0

            _tc = CONFIG["target_class"]
            _ttp, _tfp, _tfn = (train_conf[_tc]["tp"], train_conf[_tc]["fp"], train_conf[_tc]["fn"])
            train_precision = _ttp / (_ttp + _tfp) if (_ttp + _tfp) > 0 else 0.0
            train_recall = _ttp / (_ttp + _tfn) if (_ttp + _tfn) > 0 else 0.0
            train_f1 = (2 * _ttp / (2 * _ttp + _tfp + _tfn)
                       if (2 * _ttp + _tfp + _tfn) > 0 else 0.0)

            gradient_norm = float(np.mean(_grad_norms)) if _grad_norms else 0.0
            gpu_memory_used = (torch.cuda.memory_allocated() / (1024 ** 2)
                              if torch.cuda.is_available() else 0.0)
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            all_true, all_pred = [], []
            all_prob = []
            _val_item_counter = 0
            _val_per_file_conf = {f: {"tp": 0, "fp": 0, "fn": 0} for f in VAL_FILES}
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
                    _probs = torch.softmax(out.float(), dim=1)[:, CONFIG["target_class"], :]
                    all_prob.extend(_probs.cpu().numpy().ravel())

                    _bsz = x.size(0)
                    _file_idx = ((np.arange(_val_item_counter, _val_item_counter + _bsz))
                                //  CONFIG["chunks_per_cloud"])
                    _tc = CONFIG["target_class"]
                    _preds_np = preds.cpu().numpy()
                    _y_np = y.cpu().numpy()
                    for _b in range(_bsz):
                        _f = VAL_FILES[min(_file_idx[_b], len(VAL_FILES) - 1)]
                        _pv, _yv = _preds_np[_b], _y_np[_b]
                        _val_per_file_conf[_f]["tp"] += int(((_pv == _tc) & (_yv == _tc)).sum())
                        _val_per_file_conf[_f]["fp"] += int(((_pv == _tc) & (_yv != _tc)).sum())
                        _val_per_file_conf[_f]["fn"] += int(((_pv != _tc) & (_yv == _tc)).sum())
                    _val_item_counter += _bsz
            epoch_val_loss = val_loss_sum / len(val_loader.dataset)
            epoch_val_acc = val_correct / val_total
            miou = compute_miou(np.array(all_true), np.array(all_pred), NUM_CLASSES)
            dice = compute_dice(np.array(all_true), np.array(all_pred), NUM_CLASSES)
            from sklearn.metrics import (precision_score, recall_score, f1_score,
                                         roc_auc_score, average_precision_score)
            _val_true_arr = np.array(all_true)
            _val_pred_arr = np.array(all_pred)
            _val_prob_arr = np.array(all_prob)
            _val_true_bin = (_val_true_arr == CONFIG["target_class"]).astype(int)
            val_precision = precision_score(_val_true_arr, _val_pred_arr,
                                            pos_label=CONFIG["target_class"], zero_division=0)
            val_recall = recall_score(_val_true_arr, _val_pred_arr,
                                      pos_label=CONFIG["target_class"], zero_division=0)
            val_f1 = f1_score(_val_true_arr, _val_pred_arr,
                              pos_label=CONFIG["target_class"], zero_division=0)
            # if _val_true_bin.min() == _val_true_bin.max():
            #     val_roc_auc, val_pr_auc = float("nan"), float("nan")
            # else:
            #     val_roc_auc = roc_auc_score(_val_true_bin, _val_prob_arr)
            #     val_pr_auc = average_precision_score(_val_true_bin, _val_prob_arr)
            _n_bad_prob = int(~np.isfinite(_val_prob_arr)).sum() if False else int((~np.isfinite(_val_prob_arr)).sum())
            if _n_bad_prob > 0:
                log.warning(f"  {_n_bad_prob}/{_val_prob_arr.size} val probs are NaN/inf "
                            f"— model likely produced NaN logits this epoch; "
                            f"clipping for metric computation only")
                _val_prob_arr = np.nan_to_num(_val_prob_arr, nan=0.0, posinf=1.0, neginf=0.0)

            if _val_true_bin.min() == _val_true_bin.max() or _n_bad_prob == _val_prob_arr.size:
                val_roc_auc, val_pr_auc = float("nan"), float("nan")
            else:
                val_roc_auc = roc_auc_score(_val_true_bin, _val_prob_arr)
                val_pr_auc = average_precision_score(_val_true_bin, _val_prob_arr)
            log.info(f"  extra val metrics | Precision: {val_precision:.4f} | "
                     f"Recall: {val_recall:.4f} | F1: {val_f1:.4f} | "
                     f"ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f}")
            epoch_time = time.time() - _epoch_t0
            history.append({
                "epoch": epoch, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss,
                "train_acc": epoch_train_acc, "val_acc": epoch_val_acc,
                "val_miou": miou, "val_dice": dice,
                "lr": optimizer.param_groups[0]["lr"],
                "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1,
                "val_roc_auc": val_roc_auc, "val_pr_auc": val_pr_auc,
                "train_precision": train_precision, "train_recall": train_recall,
                "train_f1": train_f1, "train_miou": train_miou, "train_dice": train_dice,
                "epoch_time": epoch_time, "gradient_norm": gradient_norm,
                "gpu_memory_used": gpu_memory_used,
            })
            log.info(
                f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
                f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | "
                f"mIoU: {miou:.4f} | Dice: {dice:.4f}"
            )
            if MLFLOW_OK:
                _epoch_metrics = {
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "train_acc": epoch_train_acc,
                    "val_acc": epoch_val_acc,
                    "val_miou": miou,
                    "val_dice": dice,
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc,
                    "val_pr_auc": val_pr_auc,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "train_miou": train_miou,
                    "train_dice": train_dice,
                    "epoch_time": epoch_time,
                    "gradient_norm": gradient_norm,
                    "gpu_memory_used": gpu_memory_used,
                }
                mlflow.log_metrics(_epoch_metrics, step=epoch)
                with mlflow.start_run(run_name=f"epoch_{epoch:04d}", nested=True):
                    mlflow.log_metrics(_epoch_metrics, step=epoch)
                    _per_file_metrics = {}
                    for _f, _d in _val_per_file_conf.items():
                        _tp, _fp, _fn = _d["tp"], _d["fp"], _d["fn"]
                        _fname_key = os.path.splitext(os.path.basename(_f))[0]
                        _per_file_metrics[f"val_miou_{_fname_key}"] = (
                            _tp / (_tp + _fp + _fn) if (_tp + _fp + _fn) > 0 else 0.0)
                    mlflow.log_metrics(_per_file_metrics, step=epoch)
            scheduler.step(miou)
            if miou > best_miou + 1e-4:
                best_miou  = miou
                no_improve = 0
                torch.save({"model_state": model.state_dict(),
                            "best_val_miou": best_miou,
                            "in_channels": CONFIG["in_channels"],
                            "num_classes": NUM_CLASSES,
                            "num_points": CONFIG["num_points"],
                            "use_linearity"   : CONFIG["use_linearity"],
                            "use_planarity"   : CONFIG["use_planarity"],
                            "use_sphericity"  : CONFIG["use_sphericity"],
                            "use_verticality" : CONFIG.get("use_verticality", False),
                            "use_eigenentropy": CONFIG.get("use_eigenentropy", False),
                            "use_rgb"         : CONFIG["use_rgb"],
                            "model_hparams"   : {k: CONFIG[k] for k in CONFIG
                                                if k.startswith(("kp_", "rl_", "sp_", "pn2_", "pnx_"))},
                            "auto_reg_no_improve": auto_reg_no_improve,
                            "auto_reg_triggers"  : auto_reg_triggers}, best_path)
                log.info(f"  ✓ New best mIoU {best_miou:.4f} → saved {best_path}")
                auto_reg_no_improve = 0
            else:
                no_improve += 1
                auto_reg_no_improve += 1

            if (CONFIG.get("auto_regularize", False)
                    and auto_reg_no_improve >= CONFIG.get("auto_reg_patience", 15)
                    and auto_reg_triggers < CONFIG.get("auto_reg_max_triggers", 3)):
                auto_reg_triggers += 1
                new_wd = min(optimizer.param_groups[0]["weight_decay"]
                             * CONFIG.get("auto_reg_wd_factor", 2.0),
                             CONFIG.get("auto_reg_wd_max", 1e-2))
                for g in optimizer.param_groups:
                    g["weight_decay"] = new_wd
                n_dropout_bumped = 0
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = min(m.p + CONFIG.get("auto_reg_dropout_step", 0.05),
                                  CONFIG.get("auto_reg_dropout_max", 0.6))
                        n_dropout_bumped += 1
                auto_reg_no_improve = 0
                log.info(f"  ⚠ Auto-regularize trigger #{auto_reg_triggers}: val mIoU "
                         f"stalled {CONFIG.get('auto_reg_patience', 15)} epochs — "
                         f"weight_decay→{new_wd:.5f}, bumped {n_dropout_bumped} "
                         f"dropout layer(s) by "
                         f"{CONFIG.get('auto_reg_dropout_step', 0.05)}")

            if epoch % SAVE_EVERY == 0 or epoch == CONFIG["epochs"]:
                periodic_path = os.path.join(ckpt_dir,
                                             f"{model_name}_epoch_{epoch:04d}.pth")
                torch.save({
                    "model_state"     : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict(),
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
                    "use_verticality" : CONFIG.get("use_verticality", False),
                    "use_eigenentropy": CONFIG.get("use_eigenentropy", False),
                    "use_rgb"         : CONFIG["use_rgb"],
                    "model_hparams"   : {k: CONFIG[k] for k in CONFIG
                                        if k.startswith(("kp_", "rl_", "sp_", "pn2_", "pnx_"))},
                    "auto_reg_no_improve": auto_reg_no_improve,
                    "auto_reg_triggers"  : auto_reg_triggers,
                }, periodic_path)
                log.info(f"  💾 Periodic checkpoint saved → {periodic_path}")
                prune_old_checkpoints(keep=2)
            if no_improve >= CONFIG["patience"]:
                log.info(f"Early stop at epoch {epoch} "
                         f"(no improvement for {CONFIG['patience']} epochs)")
                break
        if MLFLOW_OK:
            mlflow.log_metric("best_val_miou", best_miou)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                epochs_h = [h["epoch"] for h in history]
                fig, axes = _plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"{model_name} — Training Diagnostics "
                            f"(overfitting shows as train/val curves diverging)",
                            fontsize=13)

                ax = axes[0, 0]
                ax.plot(epochs_h, [h["train_loss"] for h in history], label="train_loss")
                ax.plot(epochs_h, [h["val_loss"] for h in history], label="val_loss")
                ax.set_xlabel("epoch"); ax.set_ylabel("loss")
                ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

                ax = axes[0, 1]
                ax.plot(epochs_h, [h["train_acc"] for h in history], label="train_acc")
                ax.plot(epochs_h, [h["val_acc"] for h in history], label="val_acc")
                ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
                ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

                ax = axes[1, 0]
                ax.plot(epochs_h, [h["val_miou"] for h in history],
                       label="val_mIoU", color="green")
                ax.plot(epochs_h, [h["val_dice"] for h in history],
                       label="val_Dice", color="orange")
                ax.axhline(best_miou, color="red", linestyle="--",
                          label=f"best_mIoU={best_miou:.4f}")
                ax.set_xlabel("epoch"); ax.set_ylabel("score")
                ax.set_title("mIoU / Dice"); ax.legend(); ax.grid(alpha=0.3)

                ax = axes[1, 1]
                ax.plot(epochs_h, [h["lr"] for h in history], color="purple")
                ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
                ax.set_yscale("log"); ax.set_title("LR schedule")
                ax.grid(alpha=0.3)

                _plt.tight_layout()
                _diag_path = os.path.join(ckpt_dir, f"{model_name}_diagnostics.png")
                _plt.savefig(_diag_path, dpi=120)
                _plt.close(fig)
                mlflow.log_artifact(_diag_path)
            except Exception as _e:
                log.warning(f"Could not generate/log diagnostic plot: {_e}")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        log.info(f"Loaded best model (mIoU={best_miou:.4f}) from {best_path}")
    model.to("cpu")
    return model

# %%
if CV_FOLDS is None:
    model = train_model(model, MODEL_NAME)
else:
    _base_ckpt_dir = CONFIG["checkpoint_dir"]
    fold_mious = []
    for fold_idx, (TRAIN_FILES, VAL_FILES) in enumerate(CV_FOLDS):
        log.info(f"\n{'='*60}\n=== FOLD {fold_idx + 1}/{len(CV_FOLDS)} ===\n{'='*60}")
        CONFIG["checkpoint_dir"] = f"{_base_ckpt_dir}_fold{fold_idx}"
        os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
        model = build_model()
        model = train_model(model, MODEL_NAME)
        best_path = os.path.join(CONFIG["checkpoint_dir"], f"{MODEL_NAME}_best.pth")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        fold_miou = ckpt.get("best_val_miou", float("nan"))
        fold_mious.append(fold_miou)
        log.info(f"Fold {fold_idx + 1} best val mIoU: {fold_miou:.4f}")
    CONFIG["checkpoint_dir"] = _base_ckpt_dir
    fold_mious = np.array(fold_mious)
    log.info(f"\n{'='*60}")
    log.info(f"K-FOLD CV RESULTS ({CONFIG['cv_folds']} folds)")
    log.info(f"  per-fold mIoU: {[f'{m:.4f}' for m in fold_mious]}")
    log.info(f"  mean ± std   : {fold_mious.mean():.4f} ± {fold_mious.std():.4f}")
    log.info(f"{'='*60}")
    best_fold = int(np.argmax(fold_mious))
    best_fold_path = os.path.join(f"{_base_ckpt_dir}_fold{best_fold}",
                                  f"{MODEL_NAME}_best.pth")
    state = torch.load(best_fold_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(state["model_state"])
    log.info(f"Loaded fold {best_fold + 1}'s weights (best mIoU="
             f"{fold_mious[best_fold]:.4f}) into `model` for downstream use")

# %% [markdown]
# ## Visualization helper

# %%
def visualize_segmentation(points, predictions, title="Segmentation"):
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
    model.eval()
    pts, feat, lbl = get_features(path)
    n = len(feat)
    N = CONFIG["num_points"]
    B = CONFIG["batch_size"]
    log.info(f"predict_full_cloud: {n:,} points, N={N:,}/chunk, batch={B}, "
             f"device={DEVICE} — this can take a while on CPU, progress below")
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
    flush_count = 0
    def flush():
        nonlocal flush_count
        if not batch_ids:
            return
        x   = torch.from_numpy(
                  feat[np.stack(batch_ids)].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy()
        for ids, o in zip(batch_ids, out):
            np.add.at(vote_sum, (ids, o), 1)
        batch_ids.clear()
        flush_count += 1
        pct = covered.mean() * 100
        log.info(f"  flush {flush_count}: {covered.sum():,}/{n:,} pts "
                 f"covered ({pct:.1f}%)")
        import sys; sys.stdout.flush()
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

# %%
from scipy.spatial import Delaunay, cKDTree
def sor_filter(pts, k=16, std_ratio=2.0):
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

if MLFLOW_OK and test_true and all(t is not None for t in test_true):
    from sklearn.metrics import classification_report
    _all_true = np.concatenate(test_true)
    _all_pred = np.concatenate(test_pred)
    _report_dict = classification_report(
        _all_true, _all_pred, target_names=["background", "wood_powder"],
        output_dict=True, zero_division=0)
    _report_text = classification_report(
        _all_true, _all_pred, target_names=["background", "wood_powder"],
        zero_division=0)
    log.info("=== TEST CLASSIFICATION REPORT ===\n" + _report_text)

    _test_miou = compute_miou(_all_true, _all_pred, NUM_CLASSES)
    _test_dice = compute_dice(_all_true, _all_pred, NUM_CLASSES)

    try:
        with mlflow.start_run(run_id=LAST_RUN_ID):
            mlflow.log_metrics({
                "test_accuracy"          : _report_dict["accuracy"],
                "test_miou"              : _test_miou,
                "test_dice"              : _test_dice,
                "test_precision_target"  : _report_dict["wood_powder"]["precision"],
                "test_recall_target"     : _report_dict["wood_powder"]["recall"],
                "test_f1_target"         : _report_dict["wood_powder"]["f1-score"],
                "test_precision_bg"      : _report_dict["background"]["precision"],
                "test_recall_bg"         : _report_dict["background"]["recall"],
                "test_f1_bg"             : _report_dict["background"]["f1-score"],
            })
            _report_path = os.path.join(CONFIG["checkpoint_dir"],
                                        f"{MODEL_NAME}_test_classification_report.txt")
            with open(_report_path, "w") as _f:
                _f.write(_report_text)
            mlflow.log_artifact(_report_path)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt2
            from sklearn.metrics import confusion_matrix
            _cm = confusion_matrix(_all_true, _all_pred, labels=list(range(NUM_CLASSES)))
            _fig_cm, _ax_cm = _plt2.subplots(figsize=(5, 4))
            _im = _ax_cm.imshow(_cm, cmap="Blues")
            _ax_cm.set_xticks(range(NUM_CLASSES)); _ax_cm.set_yticks(range(NUM_CLASSES))
            _ax_cm.set_xticklabels(["background", "wood_powder"])
            _ax_cm.set_yticklabels(["background", "wood_powder"])
            _ax_cm.set_xlabel("Predicted"); _ax_cm.set_ylabel("True")
            _ax_cm.set_title(f"{MODEL_NAME} — Test Confusion Matrix")
            for _r in range(NUM_CLASSES):
                for _c in range(NUM_CLASSES):
                    _ax_cm.text(_c, _r, f"{_cm[_r, _c]:,}", ha="center", va="center",
                              color="white" if _cm[_r, _c] > _cm.max() / 2 else "black")
            _plt2.colorbar(_im, ax=_ax_cm)
            _plt2.tight_layout()
            _cm_path = os.path.join(CONFIG["checkpoint_dir"], f"{MODEL_NAME}_confusion_matrix.png")
            _plt2.savefig(_cm_path, dpi=120)
            _plt2.close(_fig_cm)
            mlflow.log_artifact(_cm_path)

            import csv as _csv
            _csv_path = os.path.join(CONFIG["checkpoint_dir"], f"{MODEL_NAME}_per_file_test_report.csv")
            with open(_csv_path, "w", newline="") as _f:
                _w = _csv.writer(_f)
                _w.writerow(["file", "iou", "dice", "precision", "recall",
                           "tp", "fp", "fn", "tn", "gt_wood", "pred_wood"])
                for _name, _true, _pred in zip(test_names, test_true, test_pred):
                    if _true is None:
                        continue
                    _tp = int(((_true == CONFIG["target_class"]) &
                              (_pred == CONFIG["target_class"])).sum())
                    _fp = int(((_true != CONFIG["target_class"]) &
                              (_pred == CONFIG["target_class"])).sum())
                    _fn = int(((_true == CONFIG["target_class"]) &
                              (_pred != CONFIG["target_class"])).sum())
                    _tn = int(((_true != CONFIG["target_class"]) &
                              (_pred != CONFIG["target_class"])).sum())
                    _iou = _tp / (_tp + _fp + _fn) if (_tp + _fp + _fn) > 0 else 0.0
                    _dice_v = (2 * _tp / (2 * _tp + _fp + _fn)
                              if (2 * _tp + _fp + _fn) > 0 else 0.0)
                    _prec_v = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
                    _rec_v = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
                    _gt_wood = int((_true == CONFIG["target_class"]).sum())
                    _pred_wood = int((_pred == CONFIG["target_class"]).sum())
                    _w.writerow([_name, f"{_iou:.4f}", f"{_dice_v:.4f}", f"{_prec_v:.4f}",
                               f"{_rec_v:.4f}", _tp, _fp, _fn, _tn, _gt_wood, _pred_wood])
            mlflow.log_artifact(_csv_path)
    except Exception as _e:
        log.warning(f"Could not log test metrics to MLflow (run_id={LAST_RUN_ID}): {_e}")
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
    if MLFLOW_OK:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt3
            _fig_s = _plt3.figure(figsize=(6, 5))
            _ax_s = _fig_s.add_subplot(111)
            _sub_idx = (np.random.RandomState(0).choice(len(pts), 20000, replace=False)
                       if len(pts) > 20000 else np.arange(len(pts)))
            _colors = np.where(preds[_sub_idx] == CONFIG["target_class"], "green", "red")
            _ax_s.scatter(pts[_sub_idx, 0], pts[_sub_idx, 1], c=_colors, s=1, alpha=0.5)
            _ax_s.set_title(f"{MODEL_NAME} | {name} (top-down view)")
            _ax_s.set_xlabel("X"); _ax_s.set_ylabel("Y"); _ax_s.set_aspect("equal")
            _plt3.tight_layout()
            _sample_path = os.path.join(CONFIG["checkpoint_dir"], f"{MODEL_NAME}_{name}_sample.png")
            _plt3.savefig(_sample_path, dpi=100)
            _plt3.close(_fig_s)
            with mlflow.start_run(run_id=LAST_RUN_ID):
                mlflow.log_artifact(_sample_path)
        except Exception as _e:
            log.warning(f"Could not log sample visualization for {name}: {_e}")

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



