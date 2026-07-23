# %% [markdown]
# # 🏆 Comprehensive Point Cloud Segmentation & Volume Estimation — Benchmarking Framework
# 
# A single, beginner-friendly notebook that **trains, profiles, and compares many algorithms**
# on the same data, then benchmarks **volume estimation** against ground truth, and
# automatically identifies the best methods.
# 
# | Section | What happens |
# |---|---|
# | 1 | Setup: config, hardware manager (GPU→CPU, OOM-proof), MLflow (URI-only) |
# | 2 | Data: universal loader, **automatic multi-class discovery**, splits |
# | 3 | Handcrafted point features (for classical ML & analysis) |
# | 4 | **Segmentation benchmark**: Deep Learning · Classical ML · Geometry · GNN |
# | 5 | Segmentation comparison tables + performance visualizations |
# | 6 | **Volume estimation benchmark**: Geometry · ML regression · DL/GNN regression |
# | 7 | Volume comparison vs Ground Truth + error analysis |
# | 8 | Cross-benchmark analysis, overall ranking, strengths & weaknesses |
# | 9 | Best-model auto-selection (only best models are saved) |
# | 10 | Final inference on `data/test` + interactive Open3D visualization |
# 
# **How to use (beginners):** run top-to-bottom. Everything is controlled from the single
# `CONFIG` cell — you never need to edit algorithm code. Start with `quick_mode=True`
# (small subset, few epochs) to see the whole framework work end-to-end, then set
# `quick_mode=False` for the full research-grade run.
# 

# %% [markdown]
# ## 1 · Setup — Imports, Configuration, Hardware, MLflow

# %%
# ================================================================ IMPORTS
# Required: torch numpy pandas scikit-learn scipy matplotlib joblib tqdm laspy open3d psutil mlflow
# Required for PT/GNN models: torch-cluster torch-scatter  (wheels matching your torch+CUDA)
# Optional: xgboost lightgbm catboost plyfile
from __future__ import annotations
import os, sys, gc, glob, json, time, math, copy, random, logging, warnings, tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment          # Hungarian matching
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

def _try(name):
    try:
        return __import__(name)
    except Exception:
        return None

o3d      = _try("open3d");   HAS_OPEN3D = o3d is not None
laspy    = _try("laspy")
psutil   = _try("psutil")
mlflow   = _try("mlflow");   HAS_MLFLOW = mlflow is not None
xgboost  = _try("xgboost")
lightgbm = _try("lightgbm")
catboost = _try("catboost")

# torch-cluster / torch-scatter power the transformer & GNN benchmarks
try:
    from torch_cluster import knn as tc_knn
    from torch_scatter import scatter_softmax, scatter_add, scatter_max, scatter_mean
    HAS_SCATTER = True
except Exception:
    HAS_SCATTER = False

logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("benchmark")
for lib, ok in [("open3d", HAS_OPEN3D), ("laspy", laspy is not None),
                ("mlflow", HAS_MLFLOW), ("psutil", psutil is not None),
                ("xgboost", xgboost is not None), ("lightgbm", lightgbm is not None),
                ("catboost", catboost is not None),
                ("torch-cluster/scatter", HAS_SCATTER)]:
    if not ok:
        log.warning("Optional dependency missing: %s -> the related benchmark "
                    "candidates will be skipped automatically", lib)
log.info("PyTorch %s | CUDA: %s", torch.__version__, torch.cuda.is_available())

# %%
# ================================================================ CONFIGURATION
# Every knob of the whole framework lives here — nothing is hard-coded elsewhere.
CONFIG: Dict[str, Any] = {
    # ---------------- data ----------------
    "train_dir":     "data/train",     # used for train/val/test-split + CV
    "test_dir":      "data/test",      # used ONLY for final inference (Section 10)
    "gt_volume_csv": "data/gt_volume.csv",   # columns: filename,volume (class-1 objects)
    "checkpoint_dir": "checkpoints",         # ONLY best models are written here
    "file_exts":     [".las", ".laz", ".ply", ".pcd", ".xyz", ".pts", ".txt"],
    "gt_class_original": 1,       # the ORIGINAL label code whose GT volumes you have

    # ---------------- benchmark budget ----------------
    # quick_mode=True lets you validate the entire framework in minutes.
    # For the full research run set quick_mode=False (uses all files/epochs below).
    "quick_mode":        False,
    "quick": {"max_train_files": 12, "max_val_files": 4, "max_eval_files": 4,
              "epochs": 12, "chunks_per_cloud": 2},
    "full":  {"max_train_files": None, "max_val_files": None, "max_eval_files": None,
              "epochs": 400, "chunks_per_cloud": 4},

    # ---------------- which methods to benchmark (flip to skip any) ----------------
    "enabled": {
        # deep learning segmentation
        "PointNet":False, "PointNet++":False, "DGCNN":False, "PointNeXt":False,
        "PointTransformerV1":False, "PointTransformerV2":True, "PointMLP":False,
        # classical ML segmentation
        "RandomForest":True, "ExtraTrees":True, "DecisionTree":True,
        "LogisticRegression":True, "KNN":True, "SVM":True,
        "XGBoost":True, "LightGBM":True, "CatBoost":True,
        # geometry segmentation (unsupervised baselines)
        "RANSAC-Planes":True, "EuclideanClustering":True,
        "RegionGrowing":True, "ConnectedComponents":True,
        # graph neural networks
        "GCN":False, "GraphSAGE":False, "GAT":False,
        # volume: geometry
        "vol_ConvexHull":False, "vol_AlphaShape": False, "vol_VoxelSurface":False,
        "vol_VoxelColumn":False, "vol_BallPivoting": False, "vol_Poisson": False,
        # volume: ML regression
        "vol_RF":True, "vol_ExtraTrees":True, "vol_GBoost":True, "vol_SVR":True,
        "vol_XGBoost":True, "vol_LightGBM":True, "vol_CatBoost":True,
        # volume: DL / GNN regression
        "vol_MLP":False, "vol_PointNetReg":False, "vol_PointNeXtReg":False,
        "vol_PTv1Reg": False, "vol_PTv2Reg": True,      # heavier: enable if desired
        "vol_GCNReg": False, "vol_SAGEReg": False, "vol_GATReg": False,
    },

    # ---------------- shared training hyper-parameters (DL & GNN) ----------------
    "train": {"in_channels": 7, "num_points": 81920, "batch_size": 2, "lr": 5e-4, "weight_decay": 1e-4,
              "patience": 30, "val_ratio": 0.15, "test_ratio": 0.15,   # 63-file dataset -> tighter patience
              "grad_accum": 8, "amp": True},

    # ---------------- fair evaluation protocol ----------------
    # EVERY method is scored on the SAME fixed random subset of points per eval file
    # (deep models predict the full cloud first, then are indexed at these points).
    "eval_points_per_file": 60_000,

    # ---------------- classical ML protocol ----------------
    "ml": {"train_points_per_file": 4000, "knn_features": 24, "svm_cap": 40_000},

    # ---------------- geometry segmentation ----------------
    "geo": {"ransac_dist": 0.02, "ransac_min_ratio": 0.05, "ransac_max_planes": 6,
            "dbscan_eps": 0.05, "dbscan_min_pts": 20,
            "region_normal_deg": 18.0, "region_k": 16,
            "cc_voxel": 0.05},

    # ---------------- volume estimation ----------------
    "vol": {"alpha": 0.08, "voxel": 0.02, "bpa_radii": [0.02, 0.04, 0.08],
            "poisson_depth": 8, "clean_cap": 200_000,
            "reg_cv_folds": 4, "log_target": True,
            "dl_epochs": 25, "dl_points": 2048, "dl_batch": 8, "dl_patience": 6},

    # ---------------- cross-benchmark composite ranking weights ----------------
    "rank_weights": {"miou": 0.40, "dice": 0.15, "accuracy": 0.10,
                     "train_time": 0.10, "infer_time": 0.15, "memory": 0.10},

    # ---------------- visualization ----------------
    "vis": {"show_windows": True,     # set False if Open3D windows crash your kernel
            "save_ply": False,        # per spec: no output files (flip if you want .ply)
            "max_vis_samples": 3, "window": (1280, 800)},

    # ---------------- MLflow (URI-ONLY: no local folder is ever created) ----------
    "mlflow": {"enabled": True, "uri": "http://localhost:5000",
               "experiment": "pointcloud_benchmark"},

    "resume": {"enabled": True,             # skip training if a per-algorithm
                                     # checkpoint already exists in _resume_cache/
              "recompute_eval_on_skip": False},  # True = always refresh metrics
                                                  # False = trust cached metrics (fastest)
    "seed": 42,
}
BUDGET = CONFIG["quick"] if CONFIG["quick_mode"] else CONFIG["full"]
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

def set_seed(seed):
    """Reproducibility: one seed drives python/numpy/torch (+cuda)."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(CONFIG["seed"])

# -------- hard requirement check: only the two data folders are mandatory --------
_missing = [d for d in (CONFIG["train_dir"], CONFIG["test_dir"])
            if not os.path.isdir(d)]
if _missing:
    raise RuntimeError(f"REQUIRED data folder(s) missing: {_missing}. "
                       "Create data/train and data/test, then rerun.")
log.info("Data folders OK | quick_mode=%s", CONFIG["quick_mode"])

# %% [markdown]
# ### Resume / skip-if-trained
# 
# If `CONFIG["resume"]["enabled"]` is `True` (default), every algorithm below checks
# `_resume_cache/` first. If it already trained that **exact** algorithm in a previous
# run, training is skipped and the cached weights are loaded directly — only
# new/changed algorithms are (re)trained. Delete a specific file in
# `checkpoints/_resume_cache/` (or the whole folder) to force a re-train.

# %%
# ================================================================ RESUME CACHE (per-algorithm)
# A per-algorithm checkpoint cache, separate from the final "best model only" output
# in CONFIG["checkpoint_dir"]. If CONFIG["resume"]["enabled"], every algorithm below
# first looks for an EXISTING checkpoint before training -- across THREE sources, in
# order: (1) this run's fast cache, (2) the canonical "best_seg_.../best_volume_..."
# name from Section 9, (3) a fuzzy scan of checkpoints/ for older/differently-named
# files from other notebooks (best_dgcnn_seg.pth, best_ptv1_seg.pth, etc.). A file is
# only ever adopted if it actually loads into the matching architecture -- any
# mismatch (wrong shapes/keys) is caught and the algorithm simply trains fresh, so a
# messy or ambiguous checkpoints/ folder can NEVER cause a crash or a silently wrong
# model, only a missed skip.
RESUME_DIR = os.path.join(CONFIG["checkpoint_dir"], "_resume_cache")
os.makedirs(RESUME_DIR, exist_ok=True)

def _resume_path(kind, name, ext):
    """kind: 'seg' or 'vol'. Returns a stable, filesystem-safe cache path."""
    safe = name.replace("+", "p").replace("/", "_").replace(" ", "_")
    return os.path.join(RESUME_DIR, f"{kind}_{safe}.{ext}")

def _legacy_best_path(kind, name, family, ext):
    """Canonical Section-9 'best model' filename from THIS notebook's own convention."""
    safe = name.replace("+", "p")
    if kind == "seg":
        return os.path.join(CONFIG["checkpoint_dir"], f"best_seg_{family}_{safe}.{ext}")
    return os.path.join(CONFIG["checkpoint_dir"], f"best_volume_{safe}.{ext}")

def _slug(s):
    """Lowercase, alnum-only normalization for fuzzy filename matching."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

# Ordered MOST-SPECIFIC-FIRST: once a file is claimed by an earlier (more specific)
# entry, later/shorter entries can no longer match it, so "pointnet" never steals a
# "pointnet2"/"pointnet++" file that was already claimed by the "PointNet++" entry.
# Each entry: (name, include_tokens, exclude_tokens). A filename slug matches if
# it contains ANY include token and NONE of the exclude tokens -- the excludes
# prevent substring collisions, e.g. "gcn" is contained inside "dgcnn", and
# "pointnet" is contained inside "pointnet2"/"pointnet++".
SEG_ALIASES = [
    ("PointTransformerV2", ["pointtransformerv2", "ptv2"], []),
    ("PointTransformerV1", ["pointtransformerv1", "ptv1"], []),
    ("PointNeXt",          ["pointnext"], []),
    ("PointMLP",           ["pointmlp"], []),
    ("PointNet++",         ["pointnet2", "pointnetplusplus", "pointnetpp", "pointnet++"], []),
    ("PointNet",           ["pointnet"], ["pointnet2", "pointnetplusplus", "pointnetpp"]),
    ("DGCNN",              ["dgcnn"], []),
    ("RandomForest",       ["randomforest"], []),
    ("ExtraTrees",         ["extratrees"], []),
    ("DecisionTree",       ["decisiontree"], []),
    ("LogisticRegression", ["logisticregression"], []),
    ("KNN",                ["knn"], []),
    ("SVM",                ["svm"], []),
    ("XGBoost",            ["xgboost"], []),
    ("LightGBM",           ["lightgbm"], []),
    ("CatBoost",           ["catboost"], []),
    ("GraphSAGE",          ["graphsage"], []),
    ("GCN",                ["gcn"], ["dgcnn"]),
    ("GAT",                ["gat"], []),
]
VOL_ALIASES = [
    ("vol_PTv2Reg",        ["ptv2", "pointtransformerv2"], []),
    ("vol_PTv1Reg",        ["ptv1", "pointtransformerv1"], []),
    ("vol_SAGEReg",        ["sagereg", "graphsagereg"], []),
    ("vol_GATReg",         ["gatreg"], []),
    ("vol_GCNReg",         ["gcnreg"], ["dgcnn"]),
    ("vol_PointNeXtReg",   ["pointnextreg", "pointnext"], []),
    ("vol_PointNetReg",    ["pointnetreg"], []),         # deliberately NOT "pointnet" alone
    ("vol_ExtraTrees",     ["extratrees"], []),
    ("vol_GBoost",         ["gboost", "gradientboost"], []),
    ("vol_SVR",            ["svr"], []),
    ("vol_XGBoost",        ["xgboost"], []),
    ("vol_LightGBM",       ["lightgbm"], []),
    ("vol_CatBoost",       ["catboost"], []),
    ("vol_RF",             ["volrf"], []),               # deliberately narrow: avoid
                                                         # colliding with "RandomForest"
                                                         # segmentation files
    # vol_MLP intentionally has NO fuzzy alias: generic names like
    # "volume_regressor.joblib" are too ambiguous to auto-assign safely.
]

_CLAIMED_FILES = set()   # files adopted so far this run -> never offered twice

def _fuzzy_find(kind, name, ext):
    """Scan checkpoints/ top level for an older/differently-named file that looks
    like it belongs to `name`. Returns a path or None. Never raises."""
    table = SEG_ALIASES if kind == "seg" else VOL_ALIASES
    entry = next((e for e in table if e[0] == name), None)
    if entry is None:
        return None
    tokens, excludes = entry[1], entry[2]
    try:
        entries = [f for f in os.listdir(CONFIG["checkpoint_dir"])
                  if os.path.isfile(os.path.join(CONFIG["checkpoint_dir"], f))]
    except FileNotFoundError:
        return None
    pool_key = "seg" if kind == "seg" else ("vol", "volume", "reg")
    candidates = []
    torch_exts, sk_exts = (".pt", ".pth"), (".joblib", ".pkl")
    want_exts = torch_exts if ext in ("pt", "pth") else sk_exts
    for f in entries:
        path = os.path.join(CONFIG["checkpoint_dir"], f)
        if path in _CLAIMED_FILES:
            continue
        base, real_ext = os.path.splitext(f)
        if real_ext.lower() not in want_exts:
            continue
        slug = _slug(base)
        # restrict pool: seg files vs volume files, so e.g. "..._seg.pth" is never
        # mistaken for a volume regressor and vice versa
        looks_seg = "seg" in slug
        looks_vol = any(t in slug for t in ("volume", "volreg", "reg", "vol"))
        if kind == "seg" and not looks_seg:
            continue
        if kind == "vol" and not looks_vol:
            continue
        if any(tok in slug for tok in tokens) and not any(x in slug for x in excludes):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)   # newest first
    chosen = candidates[0]
    _CLAIMED_FILES.add(chosen)
    return chosen

def find_checkpoint(kind, name, ext, family=""):
    """Returns the first usable checkpoint PATH for this exact algorithm, checked
    in order: (1) fast resume cache, (2) canonical best_seg_/best_volume_ name,
    (3) fuzzy scan of checkpoints/ for older/differently-named files. Loading is
    NOT attempted here -- callers must wrap the actual torch.load/joblib.load in
    try/except and fall back to fresh training on any failure."""
    p1 = _resume_path(kind, name, ext)
    if os.path.exists(p1):
        return p1
    p2 = _legacy_best_path(kind, name, family, ext)
    if os.path.exists(p2):
        _CLAIMED_FILES.add(p2)
        return p2
    p3 = _fuzzy_find(kind, name, ext)
    if p3:
        return p3
    return None

log.info("[resume] enabled=%s | cache dir: %s",
         CONFIG["resume"]["enabled"], RESUME_DIR)

# %%
# ================================================================ HARDWARE MANAGER
class HardwareManager:
    """Detects hardware, prevents OOM crashes, and profiles resource usage.

    Priority CUDA -> CPU. Strategies against VRAM exhaustion:
    dynamic batch-size halving, mixed precision, gradient accumulation,
    automatic cache cleanup, and a final permanent CPU fallback.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._detect()
        self.amp = bool(cfg["train"]["amp"] and self.device.type == "cuda")

    def _detect(self) -> torch.device:
        if not torch.cuda.is_available():
            log.info("[HW] no CUDA -> CPU (%d threads)", os.cpu_count() or 1)
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            return torch.device("cpu")
        free, total = torch.cuda.mem_get_info()
        log.info("[HW] CUDA %s | free %.1f/%.1f GB",
                 torch.cuda.get_device_name(0), free / 1e9, total / 1e9)
        if free / 1e9 < 1.5:
            log.warning("[HW] <1.5 GB free VRAM -> CPU fallback")
            return torch.device("cpu")
        return torch.device("cuda")

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def to_cpu(self):
        """Permanent CPU fallback (called after repeated CUDA OOM)."""
        log.warning("[HW] switching PERMANENTLY to CPU + RAM")
        self.device = torch.device("cpu")
        self.amp = False
        self.cleanup()

HW = HardwareManager(CONFIG)
DEVICE = HW.device        # convenience alias (always read HW.device after OOM events)


class Profiler:
    """Context manager: wall time + peak CPU RAM (MB) + peak GPU VRAM (MB)."""
    def __enter__(self):
        HW.cleanup()
        tracemalloc.start()
        self._rss0 = psutil.Process().memory_info().rss if psutil else 0
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rss = (psutil.Process().memory_info().rss - self._rss0) if psutil else 0
        self.cpu_mb = max(peak, rss) / 1e6
        self.gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                       if torch.cuda.is_available() else 0.0)

# %%
# ================================================================ MLFLOW (URI-ONLY)
class MLflowTracker:
    """Logs every benchmark run to the tracking SERVER at CONFIG['mlflow']['uri'].
    Never creates local files/folders. Unreachable server -> silent no-op."""

    def __init__(self, cfg):
        self.enabled = bool(cfg["mlflow"]["enabled"] and HAS_MLFLOW)
        self.active = 0
        if not self.enabled:
            log.warning("[MLflow] disabled/missing -> tracking is a no-op")
            return
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        try:
            mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
            from mlflow.tracking import MlflowClient
            MlflowClient(tracking_uri=cfg["mlflow"]["uri"]).search_experiments(max_results=1)
            mlflow.set_experiment(cfg["mlflow"]["experiment"])
            log.info("[MLflow] tracking -> %s", cfg["mlflow"]["uri"])
        except Exception as e:
            log.warning("[MLflow] server unreachable (%s) -> no-op. Start it with: "
                        "mlflow server --host 127.0.0.1 --port 5000 "
                        "--backend-store-uri sqlite:////your/store/mlflow.db", e)
            self.enabled = False

    def start(self, name, nested=False, tags=None):
        if self.enabled:
            mlflow.start_run(run_name=name, nested=nested, tags=tags or {})
            self.active += 1
    def end(self):
        if self.enabled and self.active:
            mlflow.end_run(); self.active -= 1
    def params(self, d):
        if self.enabled and self.active:
            mlflow.log_params({k: str(v)[:490] for k, v in d.items()})
    def metrics(self, d, step=None):
        if self.enabled and self.active:
            clean = {k: float(v) for k, v in d.items()
                     if v is not None and np.isfinite(float(v))}
            if clean:
                mlflow.log_metrics(clean, step=step)
    def tag(self, k, v):
        if self.enabled and self.active:
            mlflow.set_tag(k, str(v))
    def artifact(self, path):
        if self.enabled and self.active and os.path.exists(path):
            mlflow.log_artifact(path)

TRACKER = MLflowTracker(CONFIG)

# %% [markdown]
# ## 2 · Data — Universal Loader, **Automatic Multi-Class Discovery**, Splits
# 
# The loader picks the right reader from the file extension. Class handling is fully
# automatic: we scan the training labels, find every distinct class code (e.g. LAS
# classification values), and build a remap `original code → 0..K-1`. The number of
# classes, model output layers, loss, metrics, palette — everything downstream adapts
# to `NUM_CLASSES` with **no hard-coded class count anywhere**.

# %%
# ================================================================ UNIVERSAL LOADER
LABEL_KEYS = ("label", "labels", "classification", "class",
              "scalar_label", "scalar_classification", "seg", "pred")

def _load_las(path):
    assert laspy is not None, "pip install laspy for .las/.laz"
    las = laspy.read(path)
    pts = np.column_stack((np.asarray(las.x), np.asarray(las.y),
                           np.asarray(las.z))).astype(np.float64)
    lbl = None
    dims = set(las.point_format.dimension_names)
    for key in LABEL_KEYS:
        if key in dims:
            lbl = np.asarray(las[key], dtype=np.int64); break
    return pts, lbl

def _load_ply(path):
    try:
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        pts = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)
        lbl = None
        for key in LABEL_KEYS:
            if key in (v.data.dtype.names or ()):
                lbl = np.asarray(v[key], dtype=np.int64); break
        return pts, lbl
    except Exception:
        assert HAS_OPEN3D, "pip install plyfile or open3d for .ply"
        return np.asarray(o3d.io.read_point_cloud(path).points, np.float64), None

def _load_pcd(path):
    assert HAS_OPEN3D, "pip install open3d for .pcd"
    lbl = None
    try:
        tp = o3d.t.io.read_point_cloud(path)
        pts = tp.point["positions"].numpy().astype(np.float64)
        for key in LABEL_KEYS:
            if key in tp.point:
                lbl = tp.point[key].numpy().reshape(-1).astype(np.int64); break
    except Exception:
        pts = np.asarray(o3d.io.read_point_cloud(path).points, np.float64)
    return pts, lbl

def _load_text(path):
    for kw in (dict(), dict(delimiter=","), dict(skiprows=1),
               dict(delimiter=",", skiprows=1)):
        try:
            arr = np.loadtxt(path, **kw); break
        except ValueError:
            arr = None
    if arr is None:
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    arr = np.asarray(arr, np.float64)
    lbl = arr[:, 3].astype(np.int64) if arr.shape[1] >= 4 else None
    return arr[:, :3], lbl

def load_pointcloud(path):
    """(points[N,3] float64, raw labels[N] int64 or None) for any supported format."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"): return _load_las(path)
    if ext == ".ply":           return _load_ply(path)
    if ext == ".pcd":           return _load_pcd(path)
    return _load_text(path)     # .xyz .pts .txt

def list_files(folder):
    out = []
    for e in CONFIG["file_exts"]:
        out += glob.glob(os.path.join(folder, "*" + e))
    return sorted(out)

# ================================================================ AUTO CLASS DISCOVERY
TRAIN_FILES_ALL = list_files(CONFIG["train_dir"])
TEST_FILES      = list_files(CONFIG["test_dir"])
assert TRAIN_FILES_ALL, f"No supported point clouds in {CONFIG['train_dir']}"
log.info("Found %d train files, %d final-inference files",
         len(TRAIN_FILES_ALL), len(TEST_FILES))

def discover_classes(files, scan_limit=40):
    """Scan training labels -> sorted unique original codes -> remap to 0..K-1."""
    found = set()
    for f in tqdm(files[:scan_limit], desc="class discovery", leave=False):
        _, lbl = load_pointcloud(f)
        if lbl is not None:
            found |= set(np.unique(lbl).tolist())
    assert found, "No labels found in training files — segmentation needs labels."
    original = sorted(found)
    remap = {orig: i for i, orig in enumerate(original)}
    return original, remap

ORIGINAL_CODES, CLASS_REMAP = discover_classes(TRAIN_FILES_ALL)
NUM_CLASSES = len(ORIGINAL_CODES)
CLASS_NAMES = {CLASS_REMAP[o]: f"Class_{o}" for o in ORIGINAL_CODES}
GT_CLASS = CLASS_REMAP.get(CONFIG["gt_class_original"], None)  # remapped id of the GT-volume class
_remap_lut_max = max(ORIGINAL_CODES) + 1
REMAP_LUT = np.zeros(_remap_lut_max, dtype=np.int64)
for o, i in CLASS_REMAP.items():
    REMAP_LUT[o] = i
log.info("Discovered %d classes: %s (GT-volume class -> id %s)",
         NUM_CLASSES, CLASS_REMAP, GT_CLASS)

def sanitize_labels(lbl):
    """Original codes -> contiguous 0..K-1 ids; unseen/out-of-range codes -> 0."""
    if lbl is None:
        return None
    lbl = np.asarray(lbl, dtype=np.int64).copy()
    bad = (lbl < 0) | (lbl >= _remap_lut_max)
    lbl[bad] = ORIGINAL_CODES[0]
    return REMAP_LUT[lbl]

# ---------------- automatic distinguishable palette (Class0 gray, then vivid) ----------
def make_palette(k):
    base = np.array([[0.55, 0.55, 0.55], [0.90, 0.10, 0.10], [0.00, 0.75, 0.00],
                     [0.10, 0.30, 0.95], [0.95, 0.75, 0.05], [0.75, 0.10, 0.85],
                     [0.05, 0.85, 0.85], [0.95, 0.45, 0.05]])
    if k <= len(base):
        return base[:k]
    extra = plt.cm.hsv(np.linspace(0, 1, k - len(base), endpoint=False))[:, :3]
    return np.vstack([base, extra])
PALETTE = make_palette(NUM_CLASSES)
DARK_COLORS = np.array([
    # [0.35, 0.35, 0.35],   # dark gray
    [0.60, 0.05, 0.05],   # dark red
    [0.05, 0.10, 0.55],   # dark blue
    [0.45, 0.25, 0.05],   # brown
    [0.40, 0.05, 0.45],   # dark purple
    [0.05, 0.35, 0.40],   # dark teal
    [0.30, 0.30, 0.05],   # olive
])
PALETTE = np.zeros((NUM_CLASSES, 3))
_di = 0
for c in range(NUM_CLASSES):
    if c == GT_CLASS:
        PALETTE[c] = [0.0, 0.75, 0.0]                 # target -> green
    else:
        PALETTE[c] = DARK_COLORS[_di % len(DARK_COLORS)]
        _di += 1
# ================================================================ SPLITS + GT VOLUMES
rng = np.random.RandomState(CONFIG["seed"])
_perm = rng.permutation(len(TRAIN_FILES_ALL))
_nv = max(1, int(len(TRAIN_FILES_ALL) * CONFIG["train"]["val_ratio"]))
_nt = max(1, int(len(TRAIN_FILES_ALL) * CONFIG["train"]["test_ratio"]))
SPLIT = {
    "train": [TRAIN_FILES_ALL[i] for i in _perm[_nv + _nt:]][:BUDGET["max_train_files"]],
    "val":   [TRAIN_FILES_ALL[i] for i in _perm[:_nv]][:BUDGET["max_val_files"]],
    "eval":  [TRAIN_FILES_ALL[i] for i in _perm[_nv:_nv + _nt]][:BUDGET["max_eval_files"]],
}
log.info("Split -> train %d | val %d | eval(held-out test) %d",
         len(SPLIT["train"]), len(SPLIT["val"]), len(SPLIT["eval"]))

def load_gt_volumes(csv_path):
    if not os.path.exists(csv_path):
        log.warning("GT volume CSV missing (%s) -> volume benchmark limited to "
                    "geometry methods without error analysis", csv_path)
        return {}
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fcol = cols.get("filename") or cols.get("file") or df.columns[0]
    vcol = cols.get("volume") or df.columns[1]
    return {os.path.splitext(os.path.basename(str(r[fcol])))[0]: float(r[vcol])
            for _, r in df.iterrows()}
GT_VOLUMES = load_gt_volumes(CONFIG["gt_volume_csv"])
log.info("GT volumes: %d entries", len(GT_VOLUMES))

# %%
# ================================================================ DATASET & CACHE
class CloudCache:
    """Bounded in-RAM cache: normalized cloud + remapped labels + eval indices.
    (No disk cache — the framework writes nothing except best models.)"""
    def __init__(self, limit=24):
        self.limit, self._d, self._order = limit, {}, []

    def get(self, path):
        if path in self._d:
            return self._d[path]
        pts, lbl = load_pointcloud(path)
        lbl = sanitize_labels(lbl)
        center = pts.mean(axis=0, keepdims=True)
        scale = max(np.linalg.norm(pts - center, axis=1).max(), 1e-9)
        norm = ((pts - center) / scale).astype(np.float32)
        # ---- 7-channel per-point features: [norm-xyz, height-above-floor, normals]
        height = ((pts[:, 2] - pts[:, 2].min())
                  / max(pts[:, 2].max() - pts[:, 2].min(), 1e-6)).astype(np.float32)
        if HAS_OPEN3D:
            _p = o3d.geometry.PointCloud()
            _p.points = o3d.utility.Vector3dVector(pts)
            _p.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
            _p.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
            normals = np.asarray(_p.normals).astype(np.float32)
        else:
            normals = np.zeros_like(norm)
        feat = np.column_stack([norm, height[:, None], normals]).astype(np.float32)
        r = np.random.RandomState(abs(hash(os.path.basename(path))) % (2**31))
        n_eval = min(CONFIG["eval_points_per_file"], len(pts))
        entry = dict(raw=pts, norm=norm, feat=feat, lbl=lbl, center=center,
                     scale=scale,
                     eval_idx=r.choice(len(pts), n_eval, replace=False))
        self._d[path] = entry; self._order.append(path)
        if len(self._order) > self.limit:
            self._d.pop(self._order.pop(0), None)
        return entry

CACHE = CloudCache()

class ChunkDataset(Dataset):
    """Training samples for deep/GNN models: random contiguous chunks of
    `num_points` from a spatially sorted cloud -> (3,N) tensor + labels."""
    def __init__(self, files, augment=True):
        self.files, self.augment = list(files), augment
        self.cpc = BUDGET["chunks_per_cloud"]
        self.N = CONFIG["train"]["num_points"]

    def __len__(self): return len(self.files) * self.cpc

    def __getitem__(self, i):
        e = CACHE.get(self.files[i // self.cpc])
        pts, lbl = e["norm"], e["lbl"]          # xyz used for sphere-crop only
        n = len(pts)
        if n <= self.N:
            idx = np.random.choice(n, self.N, replace=True)
        else:                                   # sphere crop keeps local structure intact
            seed = np.random.randint(n)
            d2 = ((pts - pts[seed]) ** 2).sum(1)
            idx = np.argpartition(d2, self.N - 1)[:self.N]
        x, y = e["feat"][idx].copy(), lbl[idx].copy()       # (N,7)
        if self.augment:                        # rotate xyz AND normals (z-axis safe)
            th = np.random.uniform(0, 2 * np.pi); c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            x[:, :3] = x[:, :3] @ R.T * np.random.uniform(0.9, 1.1) + \
                np.random.normal(0, 0.002, (len(x), 3)).astype(np.float32)
            x[:, 4:7] = x[:, 4:7] @ R.T
        return torch.from_numpy(x.T), torch.from_numpy(y)   # (7,N), (N,)

def make_loader(files, shuffle, batch=None):
    return DataLoader(ChunkDataset(files, augment=shuffle),
                      batch_size=batch or CONFIG["train"]["batch_size"],
                      shuffle=shuffle, num_workers=0, drop_last=shuffle,
                      pin_memory=(HW.device.type == "cuda"))

def class_weights(files, sample=30):
    counts = np.zeros(NUM_CLASSES)
    for f in files[:sample]:
        counts += np.bincount(CACHE.get(f)["lbl"], minlength=NUM_CLASSES)
    w = 1.0 / np.clip(counts / counts.sum(), 1e-6, None)
    w = w / w.sum() * NUM_CLASSES
    log.info("class weights: %s", np.round(w, 3))
    return torch.tensor(w, dtype=torch.float32)

CLASS_W = class_weights(SPLIT["train"])

# %% [markdown]
# ## 3 · Handcrafted Point Features (for Classical ML)
# 
# Classical models cannot "see" raw XYZ context, so we describe each point with
# geometry features computed from its k-nearest neighbourhood: **eigen-features**
# (linearity, planarity, sphericity from the local covariance), height above the
# local floor, normal verticality, and local density. These are the standard
# features used in the point cloud classification literature.

# %%
# ================================================================ POINT FEATURES
FEATURE_NAMES = ["z_rel", "nz_abs", "linearity", "planarity", "sphericity",
                 "curvature", "density", "height_range"]

def point_features(pts_norm, idx=None):
    """Per-point handcrafted features (rows follow `idx` if given, else all points).
    Uses an Open3D KD-tree for kNN neighbourhoods; NumPy fallback for small clouds."""
    k = CONFIG["ml"]["knn_features"]
    q = pts_norm if idx is None else pts_norm[idx]
    z = pts_norm[:, 2]
    z_rel = (q[:, 2] - z.min()) / max(z.max() - z.min(), 1e-9)

    if HAS_OPEN3D and len(pts_norm) > 2000:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_norm.astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd)
        neigh = np.empty((len(q), k), np.int64)
        for i, p in enumerate(q):
            _, nn, _ = tree.search_knn_vector_3d(p.astype(np.float64), k)
            neigh[i] = np.asarray(nn)[:k]
    else:                                              # tiny clouds: brute force
        d2 = ((q[:, None, :] - pts_norm[None]) ** 2).sum(-1)
        neigh = np.argsort(d2, axis=1)[:, :k]

    nb = pts_norm[neigh]                               # (M,k,3)
    mu = nb.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb - mu, nb - mu) / k
    ev = np.linalg.eigvalsh(cov)                       # ascending l3<=l2<=l1
    l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
    s = np.clip(l1, 1e-12, None)
    linearity = (l1 - l2) / s
    planarity = (l2 - l3) / s
    sphericity = l3 / s
    curvature = l3 / np.clip(l1 + l2 + l3, 1e-12, None)
    # normal = eigenvector of smallest eigenvalue -> |z| = verticality proxy
    w, V = np.linalg.eigh(cov)
    nz_abs = np.abs(V[:, 2, 0])
    r_k = np.linalg.norm(nb[:, -1, :] - q, axis=1)     # distance to k-th neighbour
    density = k / np.clip((4 / 3) * np.pi * r_k ** 3, 1e-9, None)
    height_range = nb[:, :, 2].max(1) - nb[:, :, 2].min(1)
    X = np.column_stack([z_rel, nz_abs, linearity, planarity, sphericity,
                         curvature, np.log1p(density), height_range]).astype(np.float32)
    return np.nan_to_num(X)

# %% [markdown]
# ## 4 · Segmentation Benchmark — Common Machinery
# 
# Every method (any family) is scored **identically**: it must label the *same* fixed
# random subset of points in each held-out eval file. Deep models label the full cloud
# (chunked, OOM-safe) and are then indexed at those points. The registry below collects
# one `SegResult` per method — metrics, timings, memory — which Section 5 turns into
# comparison tables and charts. Each method is also logged as a nested MLflow run.

# %%
# ================================================================ METRICS
def confusion(gt, pred, k=None):
    k = k or NUM_CLASSES
    return np.bincount(gt * k + pred, minlength=k * k).reshape(k, k)

def seg_metrics(conf):
    """Accuracy, per-class precision/recall/F1/IoU/Dice, macro means (NaN-safe)."""
    conf = conf.astype(np.float64)
    out = {"accuracy": float(np.diag(conf).sum() / max(conf.sum(), 1))}
    ious, dices, f1s, precs, recs = [], [], [], [], []
    for c in range(conf.shape[0]):
        tp = conf[c, c]; fp = conf[:, c].sum() - tp; fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else float("nan")
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
        out[f"iou_{c}"] = float(iou); out[f"dice_{c}"] = float(dice)
        ious.append(iou); dices.append(dice); f1s.append(f1)
        precs.append(prec); recs.append(rec)
    out.update(miou=float(np.nanmean(ious)), dice=float(np.nanmean(dices)),
               f1=float(np.mean(f1s)), precision=float(np.mean(precs)),
               recall=float(np.mean(recs)))
    return out

def volume_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_pred) & np.isfinite(y_true)
    if ok.sum() < 1:
        return {m: float("nan") for m in
                ("MAE", "RMSE", "MSE", "RelErr", "PctErr", "R2")} | {"n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    rel = np.abs(yp - yt) / np.clip(np.abs(yt), 1e-9, None)
    return {"MAE": float(np.abs(yp - yt).mean()),
            "RMSE": float(np.sqrt(((yp - yt) ** 2).mean())),
            "MSE": float(((yp - yt) ** 2).mean()),
            "RelErr": float(rel.mean()), "PctErr": float(rel.mean() * 100),
            "R2": float(r2_score(yt, yp)) if ok.sum() > 1 else float("nan"),
            "n": int(ok.sum())}

# ================================================================ REGISTRY
@dataclass
class SegResult:
    name: str; family: str
    metrics: Dict[str, float] = field(default_factory=dict)
    conf: Optional[np.ndarray] = None
    train_time: float = 0.0; infer_time: float = 0.0
    gpu_mb: float = 0.0; cpu_mb: float = 0.0
    history: List[Dict] = field(default_factory=list)   # per-epoch curves (DL/GNN)
    predictor: Any = None                               # callable(entry)->labels@eval_idx
    model: Any = None                                   # for best-model saving

SEG_RESULTS: Dict[str, SegResult] = {}

def register_seg(res: SegResult):
    SEG_RESULTS[res.name] = res
    m = res.metrics
    log.info("[%s/%s] acc %.4f | mIoU %.4f | Dice %.4f | train %.1fs | infer %.2fs/file",
             res.family, res.name, m["accuracy"], m["miou"], m["dice"],
             res.train_time, res.infer_time)

def evaluate_predictor(predict_fn, files):
    """Fair protocol: labels at the fixed eval points of every eval file."""
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    t = 0.0
    for f in files:
        e = CACHE.get(f)
        t0 = time.perf_counter()
        pred = predict_fn(e)                     # labels at e["eval_idx"]
        t += time.perf_counter() - t0
        conf += confusion(e["lbl"][e["eval_idx"]], np.asarray(pred, np.int64))
    return seg_metrics(conf), conf, t / max(len(files), 1)

# %%
# ================================================================ TORCH HARNESS
# One training loop shared by every deep / GNN model:
#   AMP + gradient accumulation + dynamic batch halving on CUDA OOM
#   + permanent CPU fallback + early stopping on val mIoU. No files are written.
def train_torch_seg(model, name):
    epochs, pat = BUDGET["epochs"], CONFIG["train"]["patience"]
    batch = CONFIG["train"]["batch_size"]
    accum = max(1, CONFIG["train"]["grad_accum"])
    crit = nn.CrossEntropyLoss(weight=CLASS_W.to(HW.device))
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["train"]["lr"],
                            weight_decay=CONFIG["train"]["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=HW.amp)
    best_state, best_miou, bad, history = None, -1.0, 0, []

    def run_epoch(loader, training):
        nonlocal batch
        model.train(training)
        tot, conf = 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for step, (x, y) in enumerate(loader):
                try:
                    x, y = x.to(HW.device), y.to(HW.device)
                    with torch.autocast(HW.device.type, enabled=HW.amp):
                        logits = model(x)                       # (B,C,N)
                        loss = crit(logits, y) / accum
                    if training:
                        scaler.scale(loss).backward()
                        if (step + 1) % accum == 0:
                            scaler.step(opt); scaler.update()
                            opt.zero_grad(set_to_none=True)
                except torch.cuda.OutOfMemoryError:
                    # OOM strategy: halve batch, retry epoch; batch==1 -> CPU forever
                    HW.cleanup()
                    if batch > 1:
                        batch = max(1, batch // 2)
                        log.warning("[%s] CUDA OOM -> retrying with batch=%d", name, batch)
                    else:
                        HW.to_cpu(); model.to(HW.device)
                        crit.to(HW.device)
                        for st in opt.state.values():
                            for k2, v2 in st.items():
                                if torch.is_tensor(v2):
                                    st[k2] = v2.cpu()
                    raise _RetryEpoch()
                tot += loss.item() * accum * x.size(0)
                conf_local = confusion(y.cpu().numpy().ravel(),
                                       logits.argmax(1).detach().cpu().numpy().ravel())
                conf += conf_local
        return tot / max(len(loader.dataset), 1), seg_metrics(conf)

    class _RetryEpoch(Exception): pass

    model.to(HW.device)
    ep = 1
    while ep <= epochs:
        try:
            tr = make_loader(SPLIT["train"], True, batch)
            va = make_loader(SPLIT["val"], False, batch)
            tr_loss, tr_m = run_epoch(tr, True)
            va_loss, va_m = run_epoch(va, False)
        except _RetryEpoch:
            continue                                   # same epoch, new batch/device
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_miou": tr_m["miou"], "val_miou": va_m["miou"],
                        "val_dice": va_m["dice"], "val_acc": va_m["accuracy"]})
        TRACKER.metrics({"train_loss": tr_loss, "val_loss": va_loss,
                         "val_miou": va_m["miou"], "val_dice": va_m["dice"],
                         "val_accuracy": va_m["accuracy"]}, step=ep)
        star = ""
        if va_m["miou"] > best_miou + 1e-4:
            best_miou, bad = va_m["miou"], 0
            best_state = copy.deepcopy(model.state_dict()); star = "  *best"
        else:
            bad += 1
        log.info("[%s] ep %02d | tr %.4f | va %.4f | val mIoU %.4f%s",
                 name, ep, tr_loss, va_loss, va_m["miou"], star)
        if bad >= pat:
            log.info("[%s] early stop (no val mIoU gain for %d epochs)", name, pat)
            break
        ep += 1
    if best_state:
        model.load_state_dict(best_state)
    return model, history

@torch.no_grad()
def predict_full_cloud(model, entry, chunk=None):
    """Label EVERY point: shuffled fixed-size chunks -> batched forward -> scatter.
    Halves the chunk batch on OOM; final resort CPU."""
    model.eval(); model.to(HW.device)          # reload to GPU on demand
    N = CONFIG["train"]["num_points"]
    B = chunk or CONFIG["train"]["batch_size"]
    pts = entry["feat"]                          # (N,7) features
    n = len(pts)
    perm = np.random.RandomState(0).permutation(n)
    pad = (N - n % N) % N
    if pad:
        perm = np.concatenate([perm, perm[:pad]])
    chunks = perm.reshape(-1, N)
    out = np.zeros(n, np.int64)
    i = 0
    while i < len(chunks):
        cid = chunks[i:i + B]
        x = torch.from_numpy(pts[cid].transpose(0, 2, 1)).to(HW.device)
        try:
            pred = model(x).argmax(1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            HW.cleanup()
            if B > 1:
                B = max(1, B // 2); continue
            HW.to_cpu(); model.to(HW.device); continue
        out[cid.ravel()] = pred.ravel()
        i += B
    return out

def benchmark_torch_model(name, family, build_fn):
    """Full pipeline for one deep/GNN model: train -> eval -> profile -> register."""
    if not CONFIG["enabled"].get(name, False):
        log.info("[skip] %s disabled in CONFIG", name); return
    if family in ("DeepLearning", "GNN") and name.startswith(("PointTransformer", "G")) \
       and not HAS_SCATTER:
        log.warning("[skip] %s needs torch-cluster/torch-scatter", name); return
    cache_path = _resume_path("seg", name, "pt")
    found_path = find_checkpoint("seg", name, "pt", family) \
        if CONFIG["resume"]["enabled"] else None
    if found_path:
        try:
            ckpt = torch.load(found_path, map_location="cpu", weights_only=False)
            model = build_fn(); model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            log.warning("[resume] %s -> candidate %s did not load (%s) -> "
                       "training fresh instead", name, found_path, e)
            found_path = None
        if found_path:
            log.info("[resume] %s -> found existing checkpoint (%s), SKIPPING training",
                     name, found_path)
            def predictor(entry, _m=model):
                out = predict_full_cloud(_m, entry)[entry["eval_idx"]]
                _m.to("cpu"); HW.cleanup()
                return out
            _ckpt_complete = all(k in ckpt for k in ("metrics", "conf", "infer_time"))
            if CONFIG["resume"]["recompute_eval_on_skip"] or not _ckpt_complete:
                with Profiler() as p_ev:
                    metrics, conf, per_file = evaluate_predictor(predictor, SPLIT["eval"])
            else:
                metrics, conf, per_file = ckpt["metrics"], ckpt["conf"], ckpt["infer_time"]
            register_seg(SegResult(name=name, family=family, metrics=metrics, conf=conf,
                                   train_time=ckpt.get("train_time", 0.0), infer_time=per_file,
                                   gpu_mb=ckpt.get("gpu_mb", 0.0), cpu_mb=ckpt.get("cpu_mb", 0.0),
                                   history=ckpt.get("history", []), predictor=predictor,
                                   model=model))
            if CONFIG["resume"]["enabled"] and found_path != cache_path:
                torch.save(ckpt, cache_path)   # adopt legacy file into the fast-path cache
            return

    HW.cleanup(); set_seed(CONFIG["seed"])
    TRACKER.start(f"seg_{name}", nested=True,
                  tags={"family": family, "phase": "segmentation"})
    TRACKER.params({"model": name, "epochs": BUDGET["epochs"],
                    "num_points": CONFIG["train"]["num_points"],
                    "num_classes": NUM_CLASSES, "quick_mode": CONFIG["quick_mode"]})
    with Profiler() as p_tr:
        model = build_fn()
        model, history = train_torch_seg(model, name)
    def predictor(entry, _m=model):
        out = predict_full_cloud(_m, entry)[entry["eval_idx"]]
        _m.to("cpu"); HW.cleanup()               # release VRAM after each file batch
        return out
    with Profiler() as p_ev:
        metrics, conf, per_file = evaluate_predictor(predictor, SPLIT["eval"])
    model.to("cpu"); HW.cleanup()                # model done -> weights to RAM, VRAM free
    TRACKER.metrics({f"test_{k}": v for k, v in metrics.items()
                     if isinstance(v, float)})
    TRACKER.metrics({"train_time_s": p_tr.seconds, "infer_time_s_per_file": per_file,
                     "gpu_peak_mb": max(p_tr.gpu_mb, p_ev.gpu_mb),
                     "cpu_peak_mb": max(p_tr.cpu_mb, p_ev.cpu_mb)})
    TRACKER.end()
    register_seg(SegResult(name=name, family=family, metrics=metrics, conf=conf,
                           train_time=p_tr.seconds, infer_time=per_file,
                           gpu_mb=max(p_tr.gpu_mb, p_ev.gpu_mb),
                           cpu_mb=max(p_tr.cpu_mb, p_ev.cpu_mb),
                           history=history, predictor=predictor, model=model))
    if CONFIG["resume"]["enabled"]:
        torch.save({"model_state": model.state_dict(), "metrics": metrics, "conf": conf,
                   "train_time": p_tr.seconds, "infer_time": per_file,
                   "gpu_mb": max(p_tr.gpu_mb, p_ev.gpu_mb),
                   "cpu_mb": max(p_tr.cpu_mb, p_ev.cpu_mb), "history": history},
                  cache_path)
        log.info("[resume] %s -> cached at %s", name, cache_path)

# %% [markdown]
# ### 4a · Deep Learning Models
# 
# All seven share one interface — `model(x)` with `x (B,3,N)` → logits `(B,C,N)`, and
# `model(x, return_embedding=True)` → a global feature (reused later for DL volume
# regression). One-line intuition for each:
# 
# - **PointNet** — per-point MLP + global max-pool: fast baseline, no local context.
# - **PointNet++** — hierarchical FPS + ball-query grouping: adds local neighbourhoods.
# - **DGCNN** — EdgeConv on a *dynamically recomputed* kNN graph in feature space.
# - **PointNeXt** — PointNet++ modernized with residual InvResMLP blocks.
# - **PointTransformer V1** — vector attention over kNN (softmax over *neighbours*).
# - **PointTransformer V2** — grouped vector attention + grid pooling.
# - **PointMLP** — pure residual MLPs with a geometric affine normalization; shows how
#   far you can get *without* attention or graphs.

# %%
# ================================================================ POINTNET & POINTNET++
IN_CH = CONFIG["train"]["in_channels"]

class PointNetSeg(nn.Module):
    """PointNet: shared per-point MLPs + a global max-pooled context vector that is
    concatenated back to every point. No neighbourhood information."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(IN_CH, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv1d(1024 + 128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.4), nn.Conv1d(128, num_classes, 1))
        self.mid = nn.Sequential(nn.Conv1d(IN_CH, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.emb_dims = 1024

    def forward(self, x, return_embedding=False):
        g = self.enc(x).max(-1)[0]                       # (B,1024) global feature
        if return_embedding:
            return g
        local = self.mid(x)                              # (B,128,N)
        gexp = g.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.head(torch.cat([gexp, local], 1))    # (B,C,N)


# ---- PointNet++ utilities (shared with PointNeXt) ----
def square_distance(a, b):
    return ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)

def index_points(pts, idx):
    B = pts.shape[0]
    view = [B] + [1] * (idx.dim() - 1)
    bidx = torch.arange(B, device=pts.device).view(view).expand_as(idx)
    return pts[bidx, idx]

def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    b = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        d = ((xyz - xyz[b, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    nsample = min(nsample, N)          # FIX: never request more neighbours than points
    S = new_xyz.shape[1]
    group = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    group[square_distance(new_xyz, xyz) > radius ** 2] = N
    group = group.sort(-1)[0][:, :, :nsample]
    first = group[:, :, :1].expand(-1, -1, nsample)
    mask = group == N
    group[mask] = first[mask]
    return group

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, last = [], in_ch + 3
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU()]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feats):
        xyz_t = xyz.permute(0, 2, 1)
        fps = farthest_point_sample(xyz_t, self.npoint)
        new_xyz = index_points(xyz_t, fps)
        idx = query_ball_point(self.radius, self.nsample, xyz_t, new_xyz)
        grouped = index_points(xyz_t, idx) - new_xyz.unsqueeze(2)
        if feats is not None:
            grouped = torch.cat([grouped, index_points(feats.permute(0, 2, 1), idx)], -1)
        new_feats = self.mlp(grouped.permute(0, 3, 2, 1)).max(2)[0]
        return new_xyz.permute(0, 2, 1), new_feats

class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, last = [], in_ch
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU()]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, f1, f2):
        x1, x2 = xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1)
        d = square_distance(x1, x2)
        d, idx = d.sort(-1)
        d, idx = d[:, :, :3].clamp(min=1e-10), idx[:, :, :3]
        w = (1.0 / d); w = w / w.sum(-1, keepdim=True)
        interp = (index_points(f2.permute(0, 2, 1), idx) * w.unsqueeze(-1)).sum(2)
        out = interp.permute(0, 2, 1)
        if f1 is not None:
            out = torch.cat([f1, out], 1)
        return self.mlp(out)

class PointNet2Seg(nn.Module):
    """PointNet++ (SSG): 4 SA encoder levels + 4 FP decoder levels."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.1, 32, IN_CH, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 32, 256, [256, 256, 512])
        self.fp4 = FeaturePropagation(512 + 256, [256, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128),
                                  nn.ReLU(), nn.Dropout(0.4),
                                  nn.Conv1d(128, num_classes, 1))
        self.emb_dims = 512

    def forward(self, x, return_embedding=False):
        xyz = x[:, :3, :].contiguous()           # coordinates = first 3 channels
        l1x, l1f = self.sa1(xyz, x)              # all 7 channels as features
        l2x, l2f = self.sa2(l1x, l1f)
        l3x, l3f = self.sa3(l2x, l2f)
        l4x, l4f = self.sa4(l3x, l3f)
        if return_embedding:
            return l4f.max(-1)[0]
        l3f = self.fp4(l3x, l4x, l3f, l4f)
        l2f = self.fp3(l2x, l3x, l2f, l3f)
        l1f = self.fp2(l1x, l2x, l1f, l2f)
        l0f = self.fp1(xyz, l1x, None, l1f)
        return self.head(l0f)

# %%
# ================================================================ DGCNN · POINTNEXT · POINTMLP
def knn_graph(x, k):
    """x (B,C,N) -> kNN indices (B,N,k) in feature space (the 'dynamic' in DGCNN)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    d = -xx - inner - xx.transpose(2, 1)
    return d.topk(k, dim=-1)[1]

def edge_features(x, k):
    B, C, N = x.shape
    idx = knn_graph(x, k) + torch.arange(B, device=x.device).view(-1, 1, 1) * N
    flat = x.transpose(2, 1).reshape(B * N, C)
    nb = flat[idx.view(-1)].view(B, N, k, C)
    xc = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([nb - xc, xc], -1).permute(0, 3, 1, 2)     # (B,2C,N,k)

class DGCNNSeg(nn.Module):
    """DGCNN: stacked EdgeConv layers on dynamically recomputed feature-space graphs."""
    def __init__(self, num_classes=NUM_CLASSES, k=20):
        super().__init__()
        self.k = k
        def conv(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1), nn.BatchNorm2d(o),
                                 nn.LeakyReLU(0.2))
        self.c1, self.c2, self.c3 = conv(2 * IN_CH, 64), conv(128, 64), conv(128, 128)
        self.glob = nn.Sequential(nn.Conv1d(256, 1024, 1), nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Conv1d(1024 + 256, 256, 1), nn.BatchNorm1d(256),
                                  nn.LeakyReLU(0.2), nn.Dropout(0.4),
                                  nn.Conv1d(256, num_classes, 1))
        self.emb_dims = 1024

    def forward(self, x, return_embedding=False):
        k = min(self.k, x.shape[-1] - 1)
        x1 = self.c1(edge_features(x, k)).max(-1)[0]
        x2 = self.c2(edge_features(x1, k)).max(-1)[0]
        x3 = self.c3(edge_features(x2, k)).max(-1)[0]
        cat = torch.cat([x1, x2, x3], 1)                        # (B,256,N)
        g = self.glob(cat).max(-1)[0]                           # (B,1024)
        if return_embedding:
            return g
        gexp = g.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return self.head(torch.cat([gexp, cat], 1))


class LocalAggregation(nn.Module):
    """PointNeXt building block: ball-query aggregation at constant resolution."""
    def __init__(self, ch, radius, nsample):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.conv = nn.Sequential(nn.Conv2d(ch + 3, ch, 1), nn.BatchNorm2d(ch), nn.ReLU())

    def forward(self, xyz, feats):
        xyz_t = xyz.permute(0, 2, 1)
        idx = query_ball_point(self.radius, self.nsample, xyz_t, xyz_t)
        gx = index_points(xyz_t, idx) - xyz_t.unsqueeze(2)
        gf = index_points(feats.permute(0, 2, 1), idx)
        return self.conv(torch.cat([gx, gf], -1).permute(0, 3, 2, 1)).max(2)[0]

class InvResMLP(nn.Module):
    def __init__(self, ch, radius, nsample, exp=4):
        super().__init__()
        self.agg = LocalAggregation(ch, radius, nsample)
        self.mlp = nn.Sequential(nn.Conv1d(ch, ch * exp, 1), nn.BatchNorm1d(ch * exp),
                                 nn.ReLU(), nn.Conv1d(ch * exp, ch, 1),
                                 nn.BatchNorm1d(ch))
    def forward(self, xyz, feats):
        return F.relu(feats + self.mlp(self.agg(xyz, feats)))

class PointNeXtSeg(nn.Module):
    """PointNeXt-S: PointNet++ encoder with residual InvResMLP blocks per stage."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.1, 32, IN_CH, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 16, 256, [256, 256, 512])
        self.res = nn.ModuleList([InvResMLP(64, 0.2, 32), InvResMLP(128, 0.4, 32),
                                  InvResMLP(256, 0.8, 32), InvResMLP(512, 1.6, 16)])
        self.fp4 = FeaturePropagation(512 + 256, [256, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128),
                                  nn.ReLU(), nn.Dropout(0.4),
                                  nn.Conv1d(128, num_classes, 1))
        self.emb_dims = 512

    def forward(self, x, return_embedding=False):
        xyz = x[:, :3, :].contiguous()
        l1x, l1f = self.sa1(xyz, x);  l1f = self.res[0](l1x, l1f)
        l2x, l2f = self.sa2(l1x, l1f); l2f = self.res[1](l2x, l2f)
        l3x, l3f = self.sa3(l2x, l2f); l3f = self.res[2](l3x, l3f)
        l4x, l4f = self.sa4(l3x, l3f); l4f = self.res[3](l4x, l4f)
        if return_embedding:
            return l4f.max(-1)[0]
        l3f = self.fp4(l3x, l4x, l3f, l4f)
        l2f = self.fp3(l2x, l3x, l2f, l3f)
        l1f = self.fp2(l1x, l2x, l1f, l2f)
        return self.head(self.fp1(xyz, l1x, None, l1f))


class GeometricAffine(nn.Module):
    """PointMLP trick: normalize grouped neighbourhoods with learnable alpha/beta."""
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
    def forward(self, g):
        std = g.std(dim=(2, 3), keepdim=True) + 1e-5
        return self.alpha * (g - g.mean(dim=(2, 3), keepdim=True)) / std + self.beta

class PointMLPSeg(nn.Module):
    """PointMLP(-elite flavour): grouped points -> geometric affine -> residual MLPs.
    Demonstrates that plain MLPs + good normalization rival attention models."""
    def __init__(self, num_classes=NUM_CLASSES, k=24):
        super().__init__()
        self.k = k
        self.embed = nn.Sequential(nn.Conv1d(IN_CH, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.affine = GeometricAffine(64 + 3)
        self.local = nn.Sequential(nn.Conv2d(64 + 3, 128, 1), nn.BatchNorm2d(128),
                                   nn.ReLU())
        def res_block(ch):
            return nn.Sequential(nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch), nn.ReLU(),
                                 nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch))
        self.res1, self.res2 = res_block(128), res_block(128)
        self.glob = nn.Sequential(nn.Conv1d(128, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(512 + 128, 256, 1), nn.BatchNorm1d(256),
                                  nn.ReLU(), nn.Dropout(0.4),
                                  nn.Conv1d(256, num_classes, 1))
        self.emb_dims = 512

    def forward(self, x, return_embedding=False):
        B, _, N = x.shape
        f = self.embed(x)                                       # (B,64,N)
        xyz_t = x[:, :3, :].permute(0, 2, 1).contiguous()       # xyz for ball query
        idx = query_ball_point(0.2, min(self.k, N), xyz_t, xyz_t)
        gx = index_points(xyz_t, idx) - xyz_t.unsqueeze(2)      # (B,N,k,3)
        gf = index_points(f.permute(0, 2, 1), idx)              # (B,N,k,64)
        g = torch.cat([gx, gf], -1).permute(0, 3, 1, 2)         # (B,67,N,k)
        f = self.local(self.affine(g)).max(-1)[0]               # (B,128,N)
        f = F.relu(f + self.res1(f))
        f = F.relu(f + self.res2(f))
        g = self.glob(f).max(-1)[0]                             # (B,512)
        if return_embedding:
            return g
        gexp = g.unsqueeze(-1).expand(-1, -1, N)
        return self.head(torch.cat([gexp, f], 1))

# %%
# ================================================================ POINT TRANSFORMERS
if HAS_SCATTER:
    class PTLayer(nn.Module):
        """PTv1 vector attention: softmax over each point's k neighbours."""
        def __init__(self, ch, k=16):
            super().__init__()
            self.k = k
            self.q, self.kk, self.v = (nn.Linear(ch, ch) for _ in range(3))
            self.pos = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
            self.att = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, ch))

        def forward(self, x, pos, batch):
            e = tc_knn(pos, pos, self.k, batch, batch)
            c, nb = e[0], e[1]
            pe = self.pos(pos[c] - pos[nb])
            w = scatter_softmax(self.att(self.q(x)[c] - self.kk(x)[nb] + pe), c, dim=0)
            return scatter_add(w * (self.v(x)[nb] + pe), c, dim=0, dim_size=x.size(0))

    class PTBlock(nn.Module):
        def __init__(self, ch, k=16):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
            self.attn = PTLayer(ch, k)
            self.mlp = nn.Sequential(nn.Linear(ch, ch * 2), nn.ReLU(),
                                     nn.Linear(ch * 2, ch))
        def forward(self, x, pos, batch):
            x = x + self.attn(self.n1(x), pos, batch)
            return x + self.mlp(self.n2(x))

    class PTv1Seg(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES, k=16, dim=64):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(IN_CH, dim), nn.ReLU(),
                                       nn.Linear(dim, dim))
            self.b1 = PTBlock(dim, k)
            self.up = nn.Linear(dim, dim * 2)
            self.b2, self.b3 = PTBlock(dim * 2, k), PTBlock(dim * 2, k)
            self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 128),
                                      nn.ReLU(), nn.Dropout(0.4),
                                      nn.Linear(128, num_classes))
            self.emb_dims = dim * 4

        def _bb(self, x):
            B, C, N = x.shape
            flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
            pos = flat[:, :3].contiguous()               # xyz for kNN attention
            batch = torch.arange(B, device=x.device).repeat_interleave(N)
            h = self.embed(flat)                          # all 7 channels
            h = self.b1(h, pos, batch)
            h = F.relu(self.up(h))
            h = self.b3(self.b2(h, pos, batch), pos, batch)
            return h.view(B, N, -1)

        def forward(self, x, return_embedding=False):
            h = self._bb(x)
            if return_embedding:
                return torch.cat([h.max(1)[0], h.mean(1)], -1)
            return self.head(h).permute(0, 2, 1)

    class GVA(nn.Module):
        """PTv2 grouped vector attention with multiplier+bias position encoding."""
        def __init__(self, ch, groups=6, k=16):
            super().__init__()
            assert ch % groups == 0
            self.k, self.g, self.gc = k, groups, ch // groups
            self.q, self.kk, self.v = (nn.Linear(ch, ch) for _ in range(3))
            self.pm = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
            self.pb = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
            self.w = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, groups))

        def forward(self, x, pos, batch):
            e = tc_knn(pos, pos, self.k, batch, batch)
            c, nb = e[0], e[1]
            dp = pos[c] - pos[nb]
            pb = self.pb(dp)
            rel = (self.q(x)[c] - self.kk(x)[nb]) * self.pm(dp) + pb
            w = scatter_softmax(self.w(rel), c, dim=0)
            vg = (self.v(x)[nb] + pb).view(-1, self.g, self.gc)
            out = scatter_add(vg * w.unsqueeze(-1), c, dim=0, dim_size=x.size(0))
            return out.view(-1, self.g * self.gc)

    class PTv2Block(nn.Module):
        def __init__(self, ch, groups=6, k=16):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
            self.attn = GVA(ch, groups, k)
            self.mlp = nn.Sequential(nn.Linear(ch, ch * 2), nn.ReLU(),
                                     nn.Linear(ch * 2, ch))
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
            return xp, scatter_mean(pos, cl, dim=0), \
                   scatter_max(batch, cl, dim=0)[0], cl

    class PTv2Seg(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES, k=16, dims=(48, 96, 192), g=6):
            super().__init__()
            d = dims
            self.embed = nn.Sequential(nn.Linear(IN_CH, d[0]), nn.ReLU(),
                                       nn.Linear(d[0], d[0]))
            self.e1 = PTv2Block(d[0], g, k)
            self.p1 = GridPool(d[0], d[1], 0.08)
            self.e2 = PTv2Block(d[1], g, k)
            self.p2 = GridPool(d[1], d[2], 0.16)
            self.e3 = PTv2Block(d[2], g, k)
            self.u2 = nn.Sequential(nn.Linear(d[2] + d[1], d[1]), nn.ReLU())
            self.d2 = PTv2Block(d[1], g, k)
            self.u1 = nn.Sequential(nn.Linear(d[1] + d[0], d[0]), nn.ReLU())
            self.d1 = PTv2Block(d[0], g, k)
            self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128),
                                      nn.ReLU(), nn.Dropout(0.4),
                                      nn.Linear(128, num_classes))
            self.emb_dims = d[2] * 2

        def _bb(self, x):
            B, C, N = x.shape
            flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
            p0 = flat[:, :3].contiguous()
            b0 = torch.arange(B, device=x.device).repeat_interleave(N)
            h0 = self.e1(self.embed(flat), p0, b0)
            h1, p1, b1, c1 = self.p1(h0, p0, b0)
            h1 = self.e2(h1, p1, b1)
            h2, p2, b2, c2 = self.p2(h1, p1, b1)
            h2 = self.e3(h2, p2, b2)
            return h0, p0, b0, h1, p1, b1, c1, h2, b2, c2, B, N

        def forward(self, x, return_embedding=False):
            h0, p0, b0, h1, p1, b1, c1, h2, b2, c2, B, N = self._bb(x)
            if return_embedding:
                gmax, _ = scatter_max(h2, b2, dim=0, dim_size=B)
                return torch.cat([gmax, scatter_mean(h2, b2, dim=0, dim_size=B)], -1)
            u1 = self.d2(self.u2(torch.cat([h1, h2[c2]], 1)), p1, b1)
            u0 = self.d1(self.u1(torch.cat([h0, u1[c1]], 1)), p0, b0)
            return self.head(u0).view(B, N, -1).permute(0, 2, 1)

# %%
# ================================================================ RUN DL BENCHMARK
DL_BUILDERS = {
    "PointNet":            lambda: PointNetSeg(),
    "PointNet++":          lambda: PointNet2Seg(),
    "DGCNN":               lambda: DGCNNSeg(),
    "PointNeXt":           lambda: PointNeXtSeg(),
    "PointMLP":            lambda: PointMLPSeg(),
}
if HAS_SCATTER:
    DL_BUILDERS["PointTransformerV1"] = lambda: PTv1Seg()
    DL_BUILDERS["PointTransformerV2"] = lambda: PTv2Seg()

for _name, _build in DL_BUILDERS.items():
    benchmark_torch_model(_name, "DeepLearning", _build)

# %% [markdown]
# ### 4b · Classical Machine Learning
# 
# Each classifier learns `handcrafted features → class` on points sampled from the
# training files, then labels the eval points of the held-out files. SVM and KNN are
# capped to a subsample (they do not scale to millions of points — an honest,
# documented limitation that itself is a benchmark finding).

# %%
# ================================================================ CLASSICAL ML SEG
def build_ml_matrix(files, per_file):
    X, y = [], []
    for f in tqdm(files, desc="ML features", leave=False):
        e = CACHE.get(f)
        r = np.random.RandomState(CONFIG["seed"])
        idx = r.choice(len(e["norm"]), min(per_file, len(e["norm"])), replace=False)
        X.append(point_features(e["norm"], idx))
        y.append(e["lbl"][idx])
    return np.vstack(X), np.concatenate(y)

def ml_candidates():
    seed = CONFIG["seed"]
    c = {"RandomForest": RandomForestClassifier(200, n_jobs=-1, random_state=seed),
         "ExtraTrees": ExtraTreesClassifier(200, n_jobs=-1, random_state=seed),
         "DecisionTree": DecisionTreeClassifier(random_state=seed),
         "LogisticRegression": LogisticRegression(max_iter=500),
         "KNN": KNeighborsClassifier(15, n_jobs=-1),
         "SVM": SVC(kernel="rbf", C=1.0)}
    if xgboost:
        c["XGBoost"] = xgboost.XGBClassifier(n_estimators=300, max_depth=6,
                                             learning_rate=0.1, n_jobs=-1,
                                             random_state=seed, verbosity=0)
    if lightgbm:
        c["LightGBM"] = lightgbm.LGBMClassifier(n_estimators=300, random_state=seed,
                                                n_jobs=-1, verbose=-1)
    if catboost:
        c["CatBoost"] = catboost.CatBoostClassifier(iterations=300, depth=6,
                                                    random_seed=seed, verbose=False)
    return c

X_ml, y_ml = build_ml_matrix(SPLIT["train"], CONFIG["ml"]["train_points_per_file"])
SCALER = StandardScaler().fit(X_ml)
X_ml_s = SCALER.transform(X_ml)
# feature caches for eval files, computed ONCE and reused by every classifier (fair timing)
EVAL_FEATS = {f: SCALER.transform(point_features(CACHE.get(f)["norm"],
                                                 CACHE.get(f)["eval_idx"]))
              for f in tqdm(SPLIT["eval"], desc="eval features", leave=False)}

for _name, _clf in ml_candidates().items():
    if not CONFIG["enabled"].get(_name, False):
        continue
    def _pred(entry, _c=_clf, _f=EVAL_FEATS):
        # entries map 1:1 to files; look up the precomputed features by identity
        for kf, feats in _f.items():
            if CACHE.get(kf)["eval_idx"] is entry["eval_idx"]:
                return _c.predict(feats)
        return _c.predict(SCALER.transform(
            point_features(entry["norm"], entry["eval_idx"])))

    cache_path = _resume_path("seg", _name, "joblib")
    found_path = find_checkpoint("seg", _name, "joblib", "ClassicalML") \
        if CONFIG["resume"]["enabled"] else None
    if found_path:
        try:
            blob = joblib.load(found_path)
            _clf_loaded = blob["model"]
            _ = _clf_loaded.predict  # sanity check: must be a fitted predictor
        except Exception as e:
            log.warning("[resume] %s -> candidate %s did not load (%s) -> "
                       "training fresh instead", _name, found_path, e)
            found_path = None
        if found_path:
            log.info("[resume] %s -> found existing checkpoint (%s), SKIPPING training",
                     _name, found_path)
            _clf = _clf_loaded
            def _pred(entry, _c=_clf, _f=EVAL_FEATS):
                for kf, feats in _f.items():
                    if CACHE.get(kf)["eval_idx"] is entry["eval_idx"]:
                        return _c.predict(feats)
                return _c.predict(SCALER.transform(
                    point_features(entry["norm"], entry["eval_idx"])))
            _blob_complete = all(k in blob for k in ("metrics", "conf", "infer_time"))
            if CONFIG["resume"]["recompute_eval_on_skip"] or not _blob_complete:
                metrics, conf, per_file = evaluate_predictor(_pred, SPLIT["eval"])
            else:
                metrics, conf, per_file = blob["metrics"], blob["conf"], blob["infer_time"]
            register_seg(SegResult(name=_name, family="ClassicalML", metrics=metrics,
                                   conf=conf, train_time=blob.get("train_time", 0.0),
                                   infer_time=per_file, cpu_mb=blob.get("cpu_mb", 0.0),
                                   predictor=_pred, model=_clf))
            if CONFIG["resume"]["enabled"] and found_path != cache_path:
                joblib.dump({"model": _clf, "metrics": metrics, "conf": conf,
                            "infer_time": per_file, "train_time": blob.get("train_time", 0.0),
                            "cpu_mb": blob.get("cpu_mb", 0.0)}, cache_path)
            continue

    set_seed(CONFIG["seed"]); HW.cleanup()
    TRACKER.start(f"seg_{_name}", nested=True,
                  tags={"family": "ClassicalML", "phase": "segmentation"})
    Xtr, ytr = X_ml_s, y_ml
    if _name in ("SVM", "KNN") and len(Xtr) > CONFIG["ml"]["svm_cap"]:
        sub = np.random.RandomState(0).choice(len(Xtr), CONFIG["ml"]["svm_cap"],
                                              replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]     # documented cap: O(N^2)/O(N) methods
    with Profiler() as p_tr:
        _clf.fit(Xtr, ytr)
    with Profiler() as p_ev:
        metrics, conf, per_file = evaluate_predictor(_pred, SPLIT["eval"])
    TRACKER.params({"model": _name, "train_points": len(Xtr)})
    TRACKER.metrics({f"test_{k}": v for k, v in metrics.items()
                     if isinstance(v, float)})
    TRACKER.metrics({"train_time_s": p_tr.seconds,
                     "infer_time_s_per_file": per_file,
                     "cpu_peak_mb": max(p_tr.cpu_mb, p_ev.cpu_mb)})
    TRACKER.end()
    register_seg(SegResult(name=_name, family="ClassicalML", metrics=metrics,
                           conf=conf, train_time=p_tr.seconds, infer_time=per_file,
                           cpu_mb=max(p_tr.cpu_mb, p_ev.cpu_mb),
                           predictor=_pred, model=_clf))
    if CONFIG["resume"]["enabled"]:
        joblib.dump({"model": _clf, "metrics": metrics, "conf": conf,
                    "infer_time": per_file, "train_time": p_tr.seconds,
                    "cpu_mb": max(p_tr.cpu_mb, p_ev.cpu_mb)}, cache_path)
        log.info("[resume] %s -> cached at %s", _name, cache_path)

# %% [markdown]
# ### 4c · Geometry-Based Segmentation (Unsupervised Baselines)
# 
# These methods use *no labels at all* — they partition the cloud by geometric rules
# (planes, density clusters, smooth regions, connectivity). To score them against the
# semantic GT we assign each discovered segment to a class via **Hungarian matching**
# (maximum-overlap assignment). They are honest baselines: expect lower mIoU, but
# near-zero "training" cost.

# %%
# ================================================================ GEOMETRY SEG
def hungarian_to_classes(cluster_ids, gt):
    """Optimally map cluster ids -> class ids by maximizing overlap (Hungarian)."""
    clusters = np.unique(cluster_ids)
    inter = np.zeros((len(clusters), NUM_CLASSES))
    for i, c in enumerate(clusters):
        m = cluster_ids == c
        inter[i] = np.bincount(gt[m], minlength=NUM_CLASSES)
    rows, cols = linear_sum_assignment(-inter)
    mapping = {clusters[r]: c for r, c in zip(rows, cols)}
    for i, c in enumerate(clusters):          # unmatched clusters -> best overlap
        mapping.setdefault(c, int(inter[i].argmax()))
    return np.vectorize(mapping.get)(cluster_ids)

def geo_ransac(entry):
    """Iterative plane extraction (adaptive stop): plane i -> cluster i, rest -> last."""
    pts = entry["norm"][entry["eval_idx"]].astype(np.float64)
    if not HAS_OPEN3D:
        return np.zeros(len(pts), np.int64)
    cl = np.full(len(pts), -1, np.int64)
    remaining = np.arange(len(pts))
    g = CONFIG["geo"]
    for pi in range(g["ransac_max_planes"]):
        if len(remaining) < max(50, g["ransac_min_ratio"] * len(pts)):
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[remaining])
        try:
            _, inl = pcd.segment_plane(g["ransac_dist"], 3, 300)
        except Exception:
            break
        if len(inl) < g["ransac_min_ratio"] * len(pts):
            break
        cl[remaining[np.asarray(inl)]] = pi
        remaining = remaining[np.setdiff1d(np.arange(len(remaining)), inl,
                                           assume_unique=False)]
    cl[cl == -1] = cl.max() + 1
    return cl

def geo_dbscan(entry):
    pts = entry["norm"][entry["eval_idx"]].astype(np.float64)
    if not HAS_OPEN3D:
        return np.zeros(len(pts), np.int64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    lab = np.asarray(pcd.cluster_dbscan(CONFIG["geo"]["dbscan_eps"],
                                        CONFIG["geo"]["dbscan_min_pts"]))
    lab[lab < 0] = lab.max() + 1
    return lab

def geo_region_growing(entry):
    """Normal-smoothness BFS region growing on the eval subset."""
    pts = entry["norm"][entry["eval_idx"]].astype(np.float64)
    if not HAS_OPEN3D:
        return np.zeros(len(pts), np.int64)
    g = CONFIG["geo"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(g["region_k"]))
    nrm = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)
    cos_thr = np.cos(np.deg2rad(g["region_normal_deg"]))
    lab = np.full(len(pts), -1, np.int64)
    cur = 0
    for seed in np.argsort(-np.abs(nrm[:, 2])):        # start from flattest points
        if lab[seed] != -1:
            continue
        stack = [seed]; lab[seed] = cur
        while stack:
            p = stack.pop()
            _, nb, _ = tree.search_knn_vector_3d(pts[p], g["region_k"])
            for q in nb:
                if lab[q] == -1 and abs(np.dot(nrm[p], nrm[q])) >= cos_thr:
                    lab[q] = cur; stack.append(q)
        cur += 1
        if cur > 200:                                   # safety cap
            break
    lab[lab == -1] = cur
    return lab

def geo_connected_components(entry):
    """Voxel-graph connected components (26-connectivity via voxel hashing)."""
    pts = entry["norm"][entry["eval_idx"]]
    v = np.floor(pts / CONFIG["geo"]["cc_voxel"]).astype(np.int64)
    keys, inv = np.unique(v, axis=0, return_inverse=True)
    key_set = {tuple(k): i for i, k in enumerate(keys)}
    parent = np.arange(len(keys))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    offs = [np.array(o) for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    for i, k in enumerate(keys):
        for off in offs:
            j = key_set.get(tuple(k + off - 1))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(i) for i in range(len(keys))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[inv]

GEO_METHODS = {"RANSAC-Planes": geo_ransac, "EuclideanClustering": geo_dbscan,
               "RegionGrowing": geo_region_growing,
               "ConnectedComponents": geo_connected_components}

for _name, _fn in GEO_METHODS.items():
    if not CONFIG["enabled"].get(_name, False):
        continue
    TRACKER.start(f"seg_{_name}", nested=True,
                  tags={"family": "Geometry", "phase": "segmentation"})
    def _pred(entry, _f=_fn):
        clusters = _f(entry)
        return hungarian_to_classes(clusters, entry["lbl"][entry["eval_idx"]])
    with Profiler() as p_ev:
        metrics, conf, per_file = evaluate_predictor(_pred, SPLIT["eval"])
    TRACKER.metrics({f"test_{k}": v for k, v in metrics.items()
                     if isinstance(v, float)})
    TRACKER.metrics({"train_time_s": 0.0, "infer_time_s_per_file": per_file,
                     "cpu_peak_mb": p_ev.cpu_mb})
    TRACKER.end()
    register_seg(SegResult(name=_name, family="Geometry", metrics=metrics, conf=conf,
                           train_time=0.0, infer_time=per_file, cpu_mb=p_ev.cpu_mb,
                           predictor=_pred))

# %% [markdown]
# ### 4d · Graph Neural Networks
# 
# Points become graph nodes (features: xyz + our handcrafted descriptors would also
# work; we use xyz + learned embedding), edges come from kNN. **GCN** averages
# neighbours, **GraphSAGE** concatenates self + neighbour means, **GAT** learns
# per-edge attention. DGCNN (Section 4a) is the *dynamic*-graph member of this family
# — here graphs are static per chunk. Skipped automatically if
# `torch-cluster`/`torch-scatter` are missing.

# %%
# ================================================================ GNN SEG
if HAS_SCATTER:
    class GCNLayer(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.lin = nn.Linear(i, o)
        def forward(self, x, edge, n):
            c, nb = edge
            deg = scatter_add(torch.ones_like(c, dtype=x.dtype), c,
                              dim=0, dim_size=n).clamp(min=1)
            agg = scatter_add(x[nb], c, dim=0, dim_size=n) / deg.unsqueeze(-1)
            return self.lin(agg + x)

    class SAGELayer(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.self_lin, self.nb_lin = nn.Linear(i, o), nn.Linear(i, o)
        def forward(self, x, edge, n):
            c, nb = edge
            mean = scatter_mean(x[nb], c, dim=0, dim_size=n)
            return self.self_lin(x) + self.nb_lin(mean)

    class GATLayer(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.lin = nn.Linear(i, o)
            self.att = nn.Linear(2 * o, 1)
        def forward(self, x, edge, n):
            c, nb = edge
            h = self.lin(x)
            a = F.leaky_relu(self.att(torch.cat([h[c], h[nb]], -1)), 0.2)
            w = scatter_softmax(a, c, dim=0)
            return scatter_add(w * h[nb], c, dim=0, dim_size=n)

    class GNNSeg(nn.Module):
        """Generic 3-layer GNN segmenter on a per-chunk kNN graph."""
        def __init__(self, layer_cls, num_classes=NUM_CLASSES, k=16, h=64):
            super().__init__()
            self.k = k
            self.embed = nn.Linear(IN_CH, h)
            self.l1, self.l2, self.l3 = layer_cls(h, h), layer_cls(h, h), layer_cls(h, h)
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 64), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(64, num_classes))
            self.emb_dims = h * 2

        def _bb(self, x):
            B, C, N = x.shape
            flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
            pos = flat[:, :3].contiguous()
            batch = torch.arange(B, device=x.device).repeat_interleave(N)
            edge = tc_knn(pos, pos, self.k, batch, batch)
            h = F.relu(self.embed(flat)); n = pos.size(0)
            h = F.relu(self.l1(h, edge, n))
            h = F.relu(self.l2(h, edge, n))
            h = F.relu(self.l3(h, edge, n))
            return h.view(B, N, -1)

        def forward(self, x, return_embedding=False):
            h = self._bb(x)
            if return_embedding:
                return torch.cat([h.max(1)[0], h.mean(1)], -1)
            return self.head(h).permute(0, 2, 1)

    for _name, _layer in {"GCN": GCNLayer, "GraphSAGE": SAGELayer,
                          "GAT": GATLayer}.items():
        benchmark_torch_model(_name, "GNN",
                              lambda _l=_layer: GNNSeg(_l))
else:
    log.warning("GNN benchmarks skipped: torch-cluster/torch-scatter missing")

# %% [markdown]
# ## 5 · Segmentation Comparison — Tables & Visualizations
# 
# Master table (sorted by mIoU), per-metric winners, training curves, per-class IoU
# heatmap, resource-vs-accuracy trade-off, and confusion matrices of the leaders —
# everything a paper's benchmark section needs.

# %%
# ================================================================ SEG COMPARISON
rows = []
for r in SEG_RESULTS.values():
    row = {"method": r.name, "family": r.family, **{k: v for k, v in r.metrics.items()
           if not k.startswith(("iou_", "dice_"))},
           "train_time_s": r.train_time, "infer_s_per_file": r.infer_time,
           "gpu_MB": r.gpu_mb, "cpu_MB": r.cpu_mb}
    for c in range(NUM_CLASSES):
        row[f"IoU[{CLASS_NAMES[c]}]"] = r.metrics.get(f"iou_{c}", np.nan)
    rows.append(row)
SEG_TABLE = pd.DataFrame(rows).sort_values("miou", ascending=False).reset_index(drop=True)
print("=" * 100)
print("SEGMENTATION BENCHMARK — MASTER TABLE (sorted by mIoU)")
print("=" * 100)
display(SEG_TABLE.round(4))

print("\nPER-METRIC WINNERS")
for metric, best_low in [("accuracy", False), ("miou", False), ("dice", False),
                         ("f1", False), ("train_time_s", True),
                         ("infer_s_per_file", True)]:
    s = SEG_TABLE.set_index("method")[metric].dropna()
    if len(s):
        winner = s.idxmin() if best_low else s.idxmax()
        print(f"  {'lowest' if best_low else 'best':>7} {metric:<18}: "
              f"{winner:<22} ({s[winner]:.4f})")

FAMILY_COLORS = {"DeepLearning": "#1f77b4", "ClassicalML": "#2ca02c",
                 "Geometry": "#ff7f0e", "GNN": "#9467bd"}
colors = [FAMILY_COLORS[f] for f in SEG_TABLE["family"]]

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for a, metric, title in [(ax[0, 0], "miou", "Mean IoU"),
                         (ax[0, 1], "dice", "Dice score"),
                         (ax[1, 0], "accuracy", "Accuracy"),
                         (ax[1, 1], "f1", "Macro F1")]:
    a.barh(SEG_TABLE["method"], SEG_TABLE[metric], color=colors)
    a.set_title(title); a.invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
fig.legend(handles, FAMILY_COLORS.keys(), loc="lower center", ncol=4)
plt.suptitle("Segmentation quality by method (color = family)")
plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.show()

# ---- per-class IoU heatmap ----
iou_cols = [c for c in SEG_TABLE.columns if c.startswith("IoU[")]
plt.figure(figsize=(1.6 + 1.1 * len(iou_cols), 0.45 * len(SEG_TABLE) + 1.5))
plt.imshow(SEG_TABLE[iou_cols].values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
plt.colorbar(label="IoU")
plt.xticks(range(len(iou_cols)), [c[4:-1] for c in iou_cols], rotation=30)
plt.yticks(range(len(SEG_TABLE)), SEG_TABLE["method"])
plt.title("Per-class IoU heatmap"); plt.tight_layout(); plt.show()

# ---- accuracy vs resources trade-off ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(SEG_TABLE["train_time_s"].clip(lower=1e-2), SEG_TABLE["miou"],
              c=colors, s=70)
for _, r in SEG_TABLE.iterrows():
    ax[0].annotate(r["method"], (max(r["train_time_s"], 1e-2), r["miou"]),
                   fontsize=7, alpha=0.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("training time (s, log)")
ax[0].set_ylabel("mIoU"); ax[0].set_title("Accuracy vs training cost")
mem = SEG_TABLE[["gpu_MB", "cpu_MB"]].max(axis=1)
ax[1].scatter(SEG_TABLE["infer_s_per_file"].clip(lower=1e-3), mem, c=colors, s=70)
for (_, r), m in zip(SEG_TABLE.iterrows(), mem):
    ax[1].annotate(r["method"], (max(r["infer_s_per_file"], 1e-3), m),
                   fontsize=7, alpha=0.8)
ax[1].set_xscale("log"); ax[1].set_xlabel("inference time / file (s, log)")
ax[1].set_ylabel("peak memory (MB)"); ax[1].set_title("Speed vs memory")
plt.tight_layout(); plt.show()

# ---- training curves (DL & GNN) ----
curve_res = [r for r in SEG_RESULTS.values() if r.history]
if curve_res:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for r in curve_res:
        h = pd.DataFrame(r.history)
        ax[0].plot(h["epoch"], h["val_loss"], label=r.name)
        ax[1].plot(h["epoch"], h["val_miou"], label=r.name)
    ax[0].set_title("Validation loss"); ax[1].set_title("Validation mIoU")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.suptitle("Learning dynamics"); plt.tight_layout(); plt.show()

# ---- confusion matrices of the top-4 ----
top = SEG_TABLE.head(4)["method"].tolist()
fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 4))
axes = np.atleast_1d(axes)
for a, name in zip(axes, top):
    cm = SEG_RESULTS[name].conf.astype(float)
    cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    a.set_title(f"{name}\n(row-normalized)")
    a.set_xticks(range(NUM_CLASSES)); a.set_yticks(range(NUM_CLASSES))
    a.set_xticklabels([CLASS_NAMES[c] for c in range(NUM_CLASSES)],
                      rotation=45, fontsize=7)
    a.set_yticklabels([CLASS_NAMES[c] for c in range(NUM_CLASSES)], fontsize=7)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                   color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
plt.suptitle("Confusion matrices — top-4 by mIoU"); plt.tight_layout(); plt.show()

BEST_SEG_NAME = SEG_TABLE.iloc[0]["method"]
BEST_SEG = SEG_RESULTS[BEST_SEG_NAME]
print(f"\n>>> Best segmentation model: {BEST_SEG_NAME} "
      f"(mIoU {BEST_SEG.metrics['miou']:.4f}) — its predictions feed the "
      f"end-to-end volume benchmark below.")

# %% [markdown]
# ## 6 · Volume Estimation Benchmark
# 
# **Research-grade error decomposition.** Every volume method is evaluated on two
# inputs per eval file: (a) **GT-segmented** object points — isolates the *intrinsic*
# error of the volume method itself, and (b) points segmented by the **best model**
# from Section 5 — the honest *end-to-end* error. Comparing the two columns tells you
# how much error comes from segmentation vs the volume algorithm.
# 
# *Why geometry methods struggle:* surface-only LiDAR scans are never watertight, so
# mesh volumes (alpha shape, Poisson, Ball-Pivoting) rely on fragile hole-filling;
# convex hulls bridge concavities (systematic over-estimate); the column-filled voxel
# method assumes a pile shape (no overhangs). ML/DL regression sidesteps all of this by
# *learning the correction* from ground truth — usually the winner.

# %%
# ================================================================ VOLUME: GEOMETRY
def _clean_pts(pts, cap=None, seed=42):
    """Qhull/alpha-shape segfault guard: finite, deduped, capped points."""
    pts = np.asarray(pts, np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) > 1:
        pts = np.unique(pts, axis=0)
    cap = cap or CONFIG["vol"]["clean_cap"]
    if len(pts) > cap:
        pts = pts[np.random.RandomState(seed).choice(len(pts), cap, replace=False)]
    return pts

def vol_convex_hull(pts):
    pts = _clean_pts(pts, cap=500_000)
    if len(pts) < 4:
        return float("nan")
    try:
        return float(ConvexHull(pts, qhull_options="QJ").volume)
    except Exception:
        return float("nan")

def _mesh_volume(mesh):
    if mesh.is_watertight():
        return float(mesh.get_volume())
    vs = CONFIG["vol"]["voxel"]                      # non-watertight -> voxelized shell
    vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, vs)
    return float(len(vox.get_voxels()) * vs ** 3)

# def vol_alpha_shape(pts):
#     pts = _clean_pts(pts)
#     if not HAS_OPEN3D or len(pts) < 10:
#         return float("nan")
#     try:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts)
#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#             pcd, CONFIG["vol"]["alpha"])
#         return _mesh_volume(mesh)
#     except Exception:
#         return float("nan")

def vol_alpha_shape(pts):
    pts = _clean_pts(pts)
    if not HAS_OPEN3D or len(pts) < 10:
        return float("nan")
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(CONFIG["vol"]["voxel"])          # thin out density
        p = np.asarray(pcd.points)
        p += np.random.RandomState(0).normal(0, 1e-6, p.shape)      # break coplanarity
        pcd.points = o3d.utility.Vector3dVector(p)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, CONFIG["vol"]["alpha"])
        return _mesh_volume(mesh)
    except Exception:
        return float("nan")

def vol_voxel_surface(pts):
    """Occupied-voxel count x voxel volume (counts only the scanned shell)."""
    pts = _clean_pts(pts, cap=1_000_000)
    if len(pts) < 4:
        return float("nan")
    vs = CONFIG["vol"]["voxel"]
    vox = np.unique(np.floor((pts - pts.min(0)) / vs).astype(np.int64), axis=0)
    return float(len(vox) * vs ** 3)

def vol_voxel_column(pts):
    """Column-filled occupancy: every (x,y) column filled floor->top voxel.
    The right prior for pile-shaped objects sitting on the ground."""
    pts = _clean_pts(pts, cap=1_000_000)
    if len(pts) < 4:
        return float("nan")
    vs = CONFIG["vol"]["voxel"]
    ijk = np.floor((pts - pts.min(0)) / vs).astype(np.int64)
    key = ijk[:, 0] * 1_000_003 + ijk[:, 1]
    order = np.argsort(key)
    ks, zs = key[order], ijk[order, 2]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    col_top = np.maximum.reduceat(zs, np.r_[0, bounds])
    return float((col_top + 1).sum() * vs ** 3)

def _recon_volume(pts, kind):
    pts = _clean_pts(pts)
    if not HAS_OPEN3D or len(pts) < 30:
        return float("nan")
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(24))
        pcd.orient_normals_to_align_with_direction([0, 0, 1.0])
        if kind == "bpa":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(CONFIG["vol"]["bpa_radii"]))
        else:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=CONFIG["vol"]["poisson_depth"])
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices()
        return _mesh_volume(mesh)
    except Exception:
        return float("nan")

GEO_VOLUME_FNS = {
    "vol_ConvexHull":   vol_convex_hull,
    "vol_AlphaShape":   vol_alpha_shape,
    "vol_VoxelSurface": vol_voxel_surface,
    "vol_VoxelColumn":  vol_voxel_column,
    "vol_BallPivoting": lambda p: _recon_volume(p, "bpa"),
    "vol_Poisson":      lambda p: _recon_volume(p, "poisson"),
}

# %%
def full_cloud_labels(entry):
    """Full-cloud prediction for ANY best model: torch -> chunked GPU;
    sklearn -> handcrafted features in batches."""
    if isinstance(BEST_SEG.model, nn.Module):
        out = predict_full_cloud(BEST_SEG.model, entry)
        BEST_SEG.model.to("cpu"); HW.cleanup()   # release VRAM after each cloud
        return out
    # classical ML: compute features for all points in batches, then predict
    n = len(entry["norm"])
    out = np.empty(n, np.int64)
    for s in range(0, n, 100_000):
        idx = np.arange(s, min(s + 100_000, n))
        out[idx] = BEST_SEG.model.predict(
            SCALER.transform(point_features(entry["norm"], idx)))
    return out

# ================================================================ VOLUME: EVAL INPUTS
# Build the two point sets per eval file ONCE: GT-segmented and best-model-segmented
# object points (original coordinates, class = GT_CLASS).
assert GT_CLASS is not None, ("gt_class_original not found among discovered classes "
                              "— set CONFIG['gt_class_original'] correctly.")
VOL_EVAL = []          # dicts: name, gt_volume, pts_gt_seg, pts_pred_seg
for f in tqdm(SPLIT["eval"], desc="volume eval inputs"):
    name = os.path.splitext(os.path.basename(f))[0]
    if name not in GT_VOLUMES:
        continue
    e = CACHE.get(f)
    # full_pred = predict_full_cloud(BEST_SEG.model, e) if BEST_SEG.model is not None \
    #     else None
    # pred_lbl = full_pred if full_pred is not None else e["lbl"]
    pred_lbl = full_cloud_labels(e) if BEST_SEG.model is not None else e["lbl"]
    VOL_EVAL.append(dict(
        name=name, gt_volume=GT_VOLUMES[name],
        pts_gt_seg=e["raw"][e["lbl"] == GT_CLASS],
        pts_pred_seg=e["raw"][pred_lbl == GT_CLASS]))
log.info("Volume eval files with GT volume: %d", len(VOL_EVAL))
if not VOL_EVAL:
    log.warning("No GT volumes for the eval split -> volume benchmark will only "
                "produce raw estimates without error analysis.")

# Training set for the LEARNED volume methods (GT-segmented train/val files)
VOL_TRAIN = []
for f in tqdm(SPLIT["train"] + SPLIT["val"], desc="volume train inputs"):
    name = os.path.splitext(os.path.basename(f))[0]
    if name in GT_VOLUMES:
        e = CACHE.get(f)
        VOL_TRAIN.append(dict(name=name, gt_volume=GT_VOLUMES[name],
                              pts=e["raw"][e["lbl"] == GT_CLASS]))
log.info("Volume training samples (files with GT): %d", len(VOL_TRAIN))

# ---- shared geometric feature vector for the regression methods ----
VOL_FEATURE_NAMES = (["n_points", "bbox_vol", "ext_x", "ext_y", "ext_z", "z_mean",
                      "z_std", "spread", "hull_vol", "hull_area"]
                     + ["voxA", "voxB", "voxC"] + ["density"])
def volume_features(pts):
    f = {"n_points": float(len(pts))}
    if len(pts) < 4:
        return np.zeros(len(VOL_FEATURE_NAMES), np.float64)
    mins, maxs = pts.min(0), pts.max(0)
    ext = np.clip(maxs - mins, 1e-9, None)
    f.update(bbox_vol=float(np.prod(ext)), ext_x=float(ext[0]), ext_y=float(ext[1]),
             ext_z=float(ext[2]), z_mean=float(pts[:, 2].mean() - mins[2]),
             z_std=float(pts[:, 2].std()), spread=float(np.trace(np.cov(pts.T))))
    try:
        h = ConvexHull(_clean_pts(pts, cap=200_000), qhull_options="QJ")
        f["hull_vol"], f["hull_area"] = float(h.volume), float(h.area)
    except Exception:
        f["hull_vol"] = f["hull_area"] = 0.0
    for tag, vs in zip(("voxA", "voxB", "voxC"), (0.01, 0.02, 0.05)):
        vox = np.unique(np.floor((pts - mins) / vs).astype(np.int64), axis=0)
        f[tag] = float(len(vox) * vs ** 3)
    f["density"] = f["n_points"] / max(f["hull_vol"], 1e-9)
    return np.array([f[k] for k in VOL_FEATURE_NAMES], np.float64)

X_VOL = np.vstack([volume_features(s["pts"]) for s in VOL_TRAIN]) if VOL_TRAIN else None
Y_VOL = np.array([s["gt_volume"] for s in VOL_TRAIN]) if VOL_TRAIN else None

# %%
# ================================================================ VOLUME: REGISTRY & RUNNERS
@dataclass
class VolResult:
    name: str; family: str
    pred_gt_seg: List[float]; pred_model_seg: List[float]
    fit_time: float = 0.0; est_time: float = 0.0; cpu_mb: float = 0.0; gpu_mb: float = 0.0
VOL_RESULTS: Dict[str, VolResult] = {}

def run_volume_method(name, family, estimate_fn, fit_time=0.0, extra_prof=None):
    """Estimate every eval file with both segmentations + profile + MLflow."""
    TRACKER.start(name, nested=True, tags={"family": family, "phase": "volume"})
    with Profiler() as p:
        pg = [estimate_fn(s["pts_gt_seg"]) for s in VOL_EVAL]
        pm = [estimate_fn(s["pts_pred_seg"]) for s in VOL_EVAL]
    gt = [s["gt_volume"] for s in VOL_EVAL]
    m_end = volume_metrics(gt, pm)
    m_gt = volume_metrics(gt, pg)
    TRACKER.metrics({f"end2end_{k}": v for k, v in m_end.items()})
    TRACKER.metrics({f"gtseg_{k}": v for k, v in m_gt.items()})
    TRACKER.metrics({"fit_time_s": fit_time,
                     "estimate_time_s": p.seconds / max(len(VOL_EVAL) * 2, 1),
                     "cpu_peak_mb": p.cpu_mb, "gpu_peak_mb": p.gpu_mb})
    TRACKER.end()
    VOL_RESULTS[name] = VolResult(name, family, pg, pm, fit_time,
                                  p.seconds / max(len(VOL_EVAL) * 2, 1),
                                  p.cpu_mb, p.gpu_mb)
    log.info("[%s/%s] end-to-end MAE %.4f | with-GT-seg MAE %.4f | %.3fs/est",
             family, name, m_end["MAE"], m_gt["MAE"],
             p.seconds / max(len(VOL_EVAL) * 2, 1))

# ---- 6a geometry methods ----
for _n, _fn in GEO_VOLUME_FNS.items():
    if CONFIG["enabled"].get(_n, False) and VOL_EVAL:
        run_volume_method(_n, "Geometry", _fn)

# %%
# ================================================================ VOLUME: ML REGRESSION
def vol_reg_candidates():
    s = CONFIG["seed"]
    c = {"vol_RF": RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1,
                                         random_state=s),
         "vol_ExtraTrees": ExtraTreesRegressor(400, n_jobs=-1, random_state=s),
         "vol_GBoost": GradientBoostingRegressor(n_estimators=400,
                                                 learning_rate=0.05, random_state=s),
         "vol_SVR": SVR(kernel="rbf", C=10.0)}
    if xgboost:
        c["vol_XGBoost"] = xgboost.XGBRegressor(n_estimators=500, max_depth=5,
                                                learning_rate=0.05, n_jobs=-1,
                                                random_state=s, verbosity=0)
    if lightgbm:
        c["vol_LightGBM"] = lightgbm.LGBMRegressor(n_estimators=500, random_state=s,
                                                   n_jobs=-1, verbose=-1)
    if catboost:
        c["vol_CatBoost"] = catboost.CatBoostRegressor(iterations=500, depth=6,
                                                       random_seed=s, verbose=False)
    return c

VOL_REG_MODELS = {}
if X_VOL is not None and len(X_VOL) >= CONFIG["vol"]["reg_cv_folds"]:
    _scaler = StandardScaler().fit(X_VOL)
    Xs = _scaler.transform(X_VOL)
    yt = np.log1p(Y_VOL) if CONFIG["vol"]["log_target"] else Y_VOL
    kf = KFold(CONFIG["vol"]["reg_cv_folds"], shuffle=True,
               random_state=CONFIG["seed"])
    for _n, _est in vol_reg_candidates().items():
        if not CONFIG["enabled"].get(_n, False):
            continue
        cache_path = _resume_path("vol", _n, "joblib")
        found_path = find_checkpoint("vol", _n, "joblib") \
            if CONFIG["resume"]["enabled"] else None
        if found_path:
            try:
                blob = joblib.load(found_path)
                _est_loaded = blob["model"]
                _ = _est_loaded.predict
            except Exception as e:
                log.warning("[resume] %s -> candidate %s did not load (%s) -> "
                           "training fresh instead", _n, found_path, e)
                found_path = None
        if found_path:
            log.info("[resume] %s -> found existing regressor (%s), SKIPPING CV+fit",
                     _n, found_path)
            _est, fit_time = _est_loaded, 0.0
            if CONFIG["resume"]["enabled"] and found_path != cache_path:
                joblib.dump({"model": _est}, cache_path)
        else:
            with Profiler() as p_fit:
                maes = []
                for tr, va in kf.split(Xs):                  # honest CV on train files
                    e = copy.deepcopy(_est); e.fit(Xs[tr], yt[tr])
                    pv = e.predict(Xs[va])
                    if CONFIG["vol"]["log_target"]:
                        pv = np.expm1(pv)
                    maes.append(mean_absolute_error(Y_VOL[va], pv))
                _est.fit(Xs, yt)                             # final fit on all
            fit_time = p_fit.seconds
            log.info("    %s CV MAE (train files): %.4f", _n, np.mean(maes))
            if CONFIG["resume"]["enabled"]:
                joblib.dump({"model": _est}, cache_path)
                log.info("[resume] %s -> cached at %s", _n, cache_path)
        VOL_REG_MODELS[_n] = (_est, _scaler)
        def _estimate(pts, _m=_est, _s=_scaler):
            x = _s.transform(volume_features(pts).reshape(1, -1))
            v = float(_m.predict(x)[0])
            return max(float(np.expm1(v)) if CONFIG["vol"]["log_target"] else v, 0.0)
        if VOL_EVAL:
            run_volume_method(_n, "ML-Regression", _estimate, fit_time=fit_time)
else:
    log.warning("Too few GT-volume training files -> ML volume regression skipped")

# %%
# ================================================================ VOLUME: DL & GNN REGRESSION
def object_feat7(pts):
    """7-ch features for a segmented object: [norm-xyz, height, normals]."""
    if len(pts) < 8:
        pts = np.zeros((8, 3))
    c = pts.mean(0, keepdims=True)
    sc = max(np.linalg.norm(pts - c, axis=1).max(), 1e-9)
    pn = ((pts - c) / sc).astype(np.float32)
    h = ((pts[:, 2] - pts[:, 2].min())
         / max(pts[:, 2].max() - pts[:, 2].min(), 1e-6)).astype(np.float32)
    if HAS_OPEN3D and len(pts) > 32:
        _p = o3d.geometry.PointCloud()
        _p.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        _p.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
        _p.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
        nrm = np.asarray(_p.normals).astype(np.float32)
    else:
        nrm = np.zeros_like(pn)
    return np.column_stack([pn, h[:, None], nrm]).astype(np.float32)

class VolPointDataset(Dataset):
    """GT-class object points -> 7-ch features -> log1p(volume) target."""
    def __init__(self, items):
        self.items = items
        self.N = CONFIG["vol"]["dl_points"]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        s = self.items[i]
        if "feat7" not in s:                    # compute once, reuse across epochs
            s["feat7"] = object_feat7(s["pts"])
        f = s["feat7"]
        idx = np.random.choice(len(f), self.N, replace=len(f) < self.N)
        return torch.from_numpy(f[idx].T), torch.tensor(
            np.log1p(s["gt_volume"]), dtype=torch.float32)

class EmbRegressor(nn.Module):
    """Any segmentation backbone -> global embedding -> MLP -> log1p(volume)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(backbone.emb_dims, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        return self.head(self.backbone(x, return_embedding=True)).squeeze(-1)

def train_vol_regressor(model, name):
    v = CONFIG["vol"]
    n_val = max(1, int(0.2 * len(VOL_TRAIN)))
    tr = DataLoader(VolPointDataset(VOL_TRAIN[n_val:]), batch_size=v["dl_batch"],
                    shuffle=True, drop_last=len(VOL_TRAIN[n_val:]) > v["dl_batch"])
    va = DataLoader(VolPointDataset(VOL_TRAIN[:n_val]), batch_size=v["dl_batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()
    best, best_state, bad = np.inf, None, 0
    model.to(HW.device)
    for ep in range(1, v["dl_epochs"] + 1):
        model.train()
        for x, y in tr:
            try:
                loss = lossf(model(x.to(HW.device)), y.to(HW.device))
                opt.zero_grad(); loss.backward(); opt.step()
            except torch.cuda.OutOfMemoryError:
                HW.to_cpu(); model.to(HW.device)
                for st in opt.state.values():
                    for k2, v2 in st.items():
                        if torch.is_tensor(v2):
                            st[k2] = v2.cpu()
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for x, y in va:
                preds += np.expm1(model(x.to(HW.device)).cpu().numpy()).tolist()
                gts += np.expm1(y.numpy()).tolist()
        mae = mean_absolute_error(gts, preds) if gts else np.inf
        TRACKER.metrics({"val_mae": mae}, step=ep)
        if mae < best - 1e-6:
            best, best_state, bad = mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= v["dl_patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu"); HW.cleanup()                # training done -> VRAM free
    return model

@torch.no_grad()
def dl_estimate(model, pts):
    if len(pts) < 8:
        return float("nan")
    f = object_feat7(pts)
    idx = np.random.RandomState(0).choice(len(f), CONFIG["vol"]["dl_points"],
                                          replace=len(f) < CONFIG["vol"]["dl_points"])
    model.to(HW.device)                          # GPU on demand
    x = torch.from_numpy(f[idx].T).unsqueeze(0).to(HW.device)
    v = max(float(np.expm1(model(x).item())), 0.0)
    model.to("cpu"); HW.cleanup()                # release right after
    return v

# simple MLP on the geometric feature vector (the "MLP Regression" baseline)
class FeatMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

DL_VOL_BUILDERS = {
    "vol_MLP":          None,                                   # handled separately
    "vol_PointNetReg":  (lambda: EmbRegressor(PointNetSeg())),
    "vol_PointNeXtReg": (lambda: EmbRegressor(PointNeXtSeg())),
}
if HAS_SCATTER:
    DL_VOL_BUILDERS["vol_PTv1Reg"] = lambda: EmbRegressor(PTv1Seg())
    DL_VOL_BUILDERS["vol_PTv2Reg"] = lambda: EmbRegressor(PTv2Seg())
    DL_VOL_BUILDERS["vol_GCNReg"] = lambda: EmbRegressor(GNNSeg(GCNLayer))
    DL_VOL_BUILDERS["vol_SAGEReg"] = lambda: EmbRegressor(GNNSeg(SAGELayer))
    DL_VOL_BUILDERS["vol_GATReg"] = lambda: EmbRegressor(GNNSeg(GATLayer))

if VOL_TRAIN and VOL_EVAL:
    # ---- feature-vector MLP ----
    if CONFIG["enabled"].get("vol_MLP", False) and X_VOL is not None:
        set_seed(CONFIG["seed"])
        _s = StandardScaler().fit(X_VOL)
        mlp = FeatMLP(X_VOL.shape[1]).to(HW.device)
        cache_path = _resume_path("vol", "vol_MLP", "pt")
        found_path = find_checkpoint("vol", "vol_MLP", "pt") \
            if CONFIG["resume"]["enabled"] else None
        if found_path:
            try:
                ckpt = torch.load(found_path, map_location=HW.device, weights_only=False)
                mlp.load_state_dict(ckpt["model_state"])
            except Exception as e:
                log.warning("[resume] vol_MLP -> candidate %s did not load (%s) -> "
                           "training fresh instead", found_path, e)
                found_path = None
        if found_path:
            log.info("[resume] vol_MLP -> found existing weights (%s), SKIPPING training",
                     found_path)
            fit_time = 0.0
            if CONFIG["resume"]["enabled"] and found_path != cache_path:
                torch.save(ckpt, cache_path)
        else:
            optm = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            Xt = torch.tensor(_s.transform(X_VOL), dtype=torch.float32, device=HW.device)
            Yt = torch.tensor(np.log1p(Y_VOL), dtype=torch.float32, device=HW.device)
            with Profiler() as p_fit:
                for _ in range(400):
                    loss = F.smooth_l1_loss(mlp(Xt), Yt)
                    optm.zero_grad(); loss.backward(); optm.step()
            fit_time = p_fit.seconds
            if CONFIG["resume"]["enabled"]:
                torch.save({"model_state": mlp.state_dict()}, cache_path)
                log.info("[resume] vol_MLP -> cached at %s", cache_path)
        mlp.eval()
        def _mlp_est(pts, _m=mlp, _sc=_s):
            with torch.no_grad():
                x = torch.tensor(_sc.transform(volume_features(pts).reshape(1, -1)),
                                 dtype=torch.float32, device=HW.device)
                return max(float(np.expm1(_m(x).item())), 0.0)
        run_volume_method("vol_MLP", "DL-Regression", _mlp_est, fit_time=fit_time)
        VOL_REG_MODELS["vol_MLP"] = (mlp, _s)

    # ---- point-based DL / GNN regressors ----
    for _n, _build in DL_VOL_BUILDERS.items():
        if _build is None or not CONFIG["enabled"].get(_n, False):
            continue
        set_seed(CONFIG["seed"]); HW.cleanup()
        cache_path = _resume_path("vol", _n, "pt")
        found_path = find_checkpoint("vol", _n, "pt") \
            if CONFIG["resume"]["enabled"] else None
        if found_path:
            try:
                _reg_loaded = _build()
                ckpt = torch.load(found_path, map_location="cpu", weights_only=False)
                _reg_loaded.load_state_dict(ckpt["model_state"])
            except Exception as e:
                log.warning("[resume] %s -> candidate %s did not load (%s) -> "
                           "training fresh instead", _n, found_path, e)
                found_path = None
        if found_path:
            log.info("[resume] %s -> found existing weights (%s), SKIPPING training",
                     _n, found_path)
            _reg, fit_time = _reg_loaded, 0.0
            if CONFIG["resume"]["enabled"] and found_path != cache_path:
                torch.save(ckpt, cache_path)
        else:
            TRACKER.start(f"fit_{_n}", nested=True,
                          tags={"family": "DL-Regression", "phase": "volume_fit"})
            with Profiler() as p_fit:
                _reg = train_vol_regressor(_build(), _n)
            TRACKER.end()
            fit_time = p_fit.seconds
            if CONFIG["resume"]["enabled"]:
                torch.save({"model_state": _reg.state_dict()}, cache_path)
                log.info("[resume] %s -> cached at %s", _n, cache_path)
        fam = "GNN-Regression" if "GCN" in _n or "SAGE" in _n or "GAT" in _n \
            else "DL-Regression"
        run_volume_method(_n, fam, lambda p, _m=_reg: dl_estimate(_m, p),
                          fit_time=fit_time)
        VOL_REG_MODELS[_n] = (_reg, None)

# %% [markdown]
# ### 7·A — Per-file volume error table (testing / eval split)
# 
# One row per eval file with GT volume; one column per volume method — estimates,
# percentage errors, and the per-file winner. Uses predictions already stored in
# `VOL_RESULTS` (end-to-end, i.e. best-model segmentation) — nothing is recomputed.

# %%
# ================================================================ PER-FILE VOLUME ERROR TABLE (EVAL)
if VOL_RESULTS and VOL_EVAL:
    _files = [s["name"] for s in VOL_EVAL]
    _gt = np.array([s["gt_volume"] for s in VOL_EVAL], float)

    est = pd.DataFrame({"file": _files, "gt_volume": _gt})
    err = pd.DataFrame({"file": _files})
    for name, r in VOL_RESULTS.items():
        pred = np.asarray(r.pred_model_seg, float)
        est[name] = np.round(pred, 4)
        err[name] = np.round(np.abs(pred - _gt) / np.clip(np.abs(_gt), 1e-9, None)
                             * 100, 2)
    _mcols = list(VOL_RESULTS.keys())
    err["best_method"] = err[_mcols].astype(float).idxmin(axis=1)

    print("=" * 110)
    print("PER-FILE VOLUME ESTIMATES (end-to-end: best-seg-model points) — eval split")
    print("=" * 110)
    display(est)
    print("\nPER-FILE |%% ERROR| BY METHOD (last column = winner for that file)")
    display(err)

    summary = pd.DataFrame({
        "mean_pct_err":   err[_mcols].mean().round(2),
        "median_pct_err": err[_mcols].median().round(2),
        "worst_pct_err":  err[_mcols].max().round(2),
        "files_won":      err["best_method"].value_counts()
                             .reindex(_mcols).fillna(0).astype(int),
    }).sort_values("mean_pct_err")
    print("\nSUMMARY ACROSS EVAL FILES")
    display(summary)

    # ---- heatmap: methods x files ----
    order = summary.index.tolist()
    M = err.set_index("file")[order].T.values.astype(float)
    plt.figure(figsize=(1.8 + 1.1 * len(_files), 0.42 * len(order) + 1.6))
    vmax = np.nanpercentile(M, 95) if np.isfinite(M).any() else 1.0
    plt.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=max(vmax, 1e-6))
    plt.colorbar(label="|% error|")
    plt.xticks(range(len(_files)), _files, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(order)), order, fontsize=8)
    for r_i in range(M.shape[0]):
        for c_i in range(M.shape[1]):
            if np.isfinite(M[r_i, c_i]):
                plt.text(c_i, r_i, f"{M[r_i, c_i]:.0f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if M[r_i, c_i] > vmax * 0.6 else "black")
    plt.title("Per-file |% error| — every volume method × every eval file "
              "(green = accurate)")
    plt.tight_layout(); plt.show()
else:
    log.warning("VOL_RESULTS/VOL_EVAL empty -> per-file error table skipped")

# %% [markdown]
# ## 7 · Volume Comparison vs Ground Truth — Error Analysis
# 
# Tables + four diagnostic plots: metric bars, predicted-vs-GT scatter (points on the
# diagonal are perfect), relative-error distributions, and the **segmentation-impact**
# chart (end-to-end error minus GT-segmentation error = how much the segmenter costs
# each volume method).

# %%
# ================================================================ VOLUME COMPARISON
if VOL_RESULTS:
    gt = np.array([s["gt_volume"] for s in VOL_EVAL])
    vrows = []
    for r in VOL_RESULTS.values():
        m_end = volume_metrics(gt, r.pred_model_seg)
        m_gts = volume_metrics(gt, r.pred_gt_seg)
        vrows.append({"method": r.name, "family": r.family,
                      **{k: m_end[k] for k in ("MAE", "RMSE", "MSE", "RelErr",
                                               "PctErr", "R2")},
                      "MAE_gtseg": m_gts["MAE"], "PctErr_gtseg": m_gts["PctErr"],
                      "seg_cost_MAE": m_end["MAE"] - m_gts["MAE"],
                      "fit_time_s": r.fit_time, "est_time_s": r.est_time,
                      "failures": int(np.sum(~np.isfinite(r.pred_model_seg)))})
    VOL_TABLE = pd.DataFrame(vrows).sort_values("MAE").reset_index(drop=True)
    print("=" * 100)
    print("VOLUME BENCHMARK — end-to-end (predicted segmentation) vs GT volumes")
    print("=" * 100)
    display(VOL_TABLE.round(4))

    fam_col = {"Geometry": "#ff7f0e", "ML-Regression": "#2ca02c",
               "DL-Regression": "#1f77b4", "GNN-Regression": "#9467bd"}
    vcolors = [fam_col.get(f, "gray") for f in VOL_TABLE["family"]]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].barh(VOL_TABLE["method"], VOL_TABLE["MAE"], color=vcolors)
    ax[0].set_title("MAE (lower = better)"); ax[0].invert_yaxis()
    ax[1].barh(VOL_TABLE["method"], VOL_TABLE["PctErr"], color=vcolors)
    ax[1].set_title("Mean % error"); ax[1].invert_yaxis()
    ax[2].barh(VOL_TABLE["method"], VOL_TABLE["R2"].clip(-1), color=vcolors)
    ax[2].set_title("R² (higher = better)"); ax[2].invert_yaxis()
    plt.suptitle("Volume estimation quality — end-to-end")
    plt.tight_layout(); plt.show()

    # ---- predicted vs GT scatter grid ----
    top_v = VOL_TABLE.head(min(6, len(VOL_TABLE)))["method"]
    ncol = 3
    nrow = int(np.ceil(len(top_v) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow))
    for a, name in zip(np.ravel(axes), top_v):
        r = VOL_RESULTS[name]
        a.scatter(gt, r.pred_model_seg, s=45, label="pred-seg", alpha=0.85)
        a.scatter(gt, r.pred_gt_seg, s=45, marker="x", label="GT-seg", alpha=0.85)
        lim = [0, max(np.nanmax(gt), np.nanmax(r.pred_model_seg)) * 1.05]
        a.plot(lim, lim, "k--", lw=1)
        a.set_title(name, fontsize=9)
        a.set_xlabel("GT volume"); a.set_ylabel("estimated")
        a.legend(fontsize=7)
    for a in np.ravel(axes)[len(top_v):]:
        a.axis("off")
    plt.suptitle("Predicted vs ground-truth volume (diagonal = perfect)")
    plt.tight_layout(); plt.show()

    # ---- relative-error distribution + segmentation impact ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    box_data, labels = [], []
    for name in VOL_TABLE["method"]:
        r = VOL_RESULTS[name]
        rel = np.abs(np.array(r.pred_model_seg) - gt) / np.clip(np.abs(gt), 1e-9, None)
        box_data.append(rel[np.isfinite(rel)] * 100); labels.append(name)
    try:
        ax[0].boxplot(box_data, orientation="horizontal", tick_labels=labels,
                      showfliers=True)
    except TypeError:                       # matplotlib < 3.9
        ax[0].boxplot(box_data, vert=False, labels=labels, showfliers=True)
    ax[0].set_xlabel("|relative error| (%)"); ax[0].set_title("Error distributions")
    ax[1].barh(VOL_TABLE["method"], VOL_TABLE["seg_cost_MAE"], color=vcolors)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_title("Segmentation cost = end-to-end MAE − GT-seg MAE")
    ax[1].invert_yaxis()
    plt.tight_layout(); plt.show()

    BEST_VOL_NAME = VOL_TABLE.iloc[0]["method"]
    print(f">>> Best volume method (end-to-end MAE): {BEST_VOL_NAME}")
else:
    VOL_TABLE, BEST_VOL_NAME = pd.DataFrame(), None
    log.warning("No volume results (no GT volumes?) -> Sections 7-8 volume parts skipped")

# %% [markdown]
# ## 8 · Cross-Benchmark Analysis — Overall Ranking, Strengths & Weaknesses
# 
# The composite segmentation score min-max-normalizes each criterion and applies the
# weights from `CONFIG["rank_weights"]` (accuracy-dominant by default — edit them to
# match *your* priorities, e.g. an embedded deployment would upweight memory).

# %%
# ================================================================ CROSS ANALYSIS
def _norm(v, invert=False):
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.nanmax(v[np.isfinite(v)]) * 2
                 if np.isfinite(v).any() else 1.0)
    rng = v.max() - v.min()
    n = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)
    return 1 - n if invert else n

w = CONFIG["rank_weights"]
mem_all = SEG_TABLE[["gpu_MB", "cpu_MB"]].max(axis=1)
SEG_TABLE["composite"] = (
    w["miou"] * _norm(SEG_TABLE["miou"])
    + w["dice"] * _norm(SEG_TABLE["dice"])
    + w["accuracy"] * _norm(SEG_TABLE["accuracy"])
    + w["train_time"] * _norm(SEG_TABLE["train_time_s"], invert=True)
    + w["infer_time"] * _norm(SEG_TABLE["infer_s_per_file"], invert=True)
    + w["memory"] * _norm(mem_all, invert=True))
RANKING = SEG_TABLE.sort_values("composite", ascending=False).reset_index(drop=True)
print("OVERALL SEGMENTATION RANKING (composite of quality + speed + memory)")
display(RANKING[["method", "family", "miou", "dice", "accuracy",
                 "train_time_s", "infer_s_per_file", "composite"]].round(4))

print("\nBEST-OF SUMMARY")
best_of = {
    "Best accuracy": SEG_TABLE.loc[SEG_TABLE["accuracy"].idxmax(), "method"],
    "Best IoU/mIoU": SEG_TABLE.loc[SEG_TABLE["miou"].idxmax(), "method"],
    "Best Dice": SEG_TABLE.loc[SEG_TABLE["dice"].idxmax(), "method"],
    "Fastest training": SEG_TABLE.loc[SEG_TABLE["train_time_s"].idxmin(), "method"],
    "Fastest inference": SEG_TABLE.loc[SEG_TABLE["infer_s_per_file"].idxmin(), "method"],
    "Lowest memory": SEG_TABLE.loc[mem_all.idxmin(), "method"],
    "Overall champion": RANKING.iloc[0]["method"],
}
if len(VOL_TABLE):
    best_of |= {
        "Best volume MAE": VOL_TABLE.loc[VOL_TABLE["MAE"].idxmin(), "method"],
        "Best volume RMSE": VOL_TABLE.loc[VOL_TABLE["RMSE"].idxmin(), "method"],
    }
    if VOL_TABLE["R2"].notna().any():          # guard: R2 can be all-NaN when too
        best_of["Best volume R²"] = VOL_TABLE.loc[VOL_TABLE["R2"].idxmax(), "method"]
    else:
        log.warning("R2 is undefined for every volume method (too few GT-volume "
                    "eval files) -> 'Best volume R²' skipped")
for k, v in best_of.items():
    print(f"  {k:<20}: {v}")

# ---- strengths & weaknesses: curated notes + measured evidence ----
NOTES = {
 "PointNet": "Simplest & fastest DL; no local context -> struggles at class borders.",
 "PointNet++": "Hierarchical neighbourhoods; solid all-rounder; FPS makes it slower.",
 "DGCNN": "Dynamic feature-space graphs capture fine structure; memory-hungry (kNN each layer).",
 "PointNeXt": "PointNet++ + residual blocks; usually the best accuracy/cost trade-off.",
 "PointTransformerV1": "Vector attention = strong context; kNN attention costs time.",
 "PointTransformerV2": "Grouped attention + grid pooling; scales better than V1.",
 "PointMLP": "Pure MLPs + affine norm; surprisingly strong, very portable.",
 "RandomForest": "Robust on handcrafted features; no spatial learning beyond them.",
 "ExtraTrees": "Like RF with more randomness; fast, slightly noisier.",
 "DecisionTree": "Interpretable but overfits; baseline only.",
 "LogisticRegression": "Linear boundary on features; fast sanity baseline.",
 "KNN": "No training but slow inference; capped subsample here.",
 "SVM": "Strong on small data; O(N²) -> capped subsample, not scalable.",
 "XGBoost": "Boosted trees; typically the best classical performer.",
 "LightGBM": "Fastest boosted trees at similar accuracy.",
 "CatBoost": "Boosting with strong defaults; slower to train.",
 "RANSAC-Planes": "Finds planar structure (floors/walls) without labels; semantic-blind.",
 "EuclideanClustering": "Density clusters; depends heavily on eps; semantic-blind.",
 "RegionGrowing": "Smooth-surface regions; sensitive to normal quality.",
 "ConnectedComponents": "Cheapest spatial partition; only separates disjoint objects.",
 "GCN": "Mean aggregation; smooths across class borders.",
 "GraphSAGE": "Self+neighbour split resists over-smoothing; good speed.",
 "GAT": "Learned edge attention; the most expressive & most expensive GNN here.",
}
print("\nSTRENGTHS & WEAKNESSES (with measured evidence)")
for _, r in RANKING.iterrows():
    note = NOTES.get(r["method"], "")
    print(f"  • {r['method']:<20} [{r['family']:<11}] mIoU {r['miou']:.3f}, "
          f"train {r['train_time_s']:.1f}s, infer {r['infer_s_per_file']:.2f}s/file — {note}")

# %% [markdown]
# ## 9 · Best-Model Auto-Selection & Saving
# 
# Per the framework rules, **only the best model of each family** is written to
# `checkpoints/` — no logs, no CSVs, no caches, no intermediate checkpoints.

# %%
# ================================================================ SAVE BEST MODELS
saved = []
for family in ("DeepLearning", "GNN"):
    fam = [r for r in SEG_RESULTS.values() if r.family == family and r.model is not None]
    if fam:
        best = max(fam, key=lambda r: r.metrics["miou"])
        path = os.path.join(CONFIG["checkpoint_dir"],
                            f"best_seg_{family}_{best.name}.pth".replace("+", "p"))
        torch.save({"model_state": best.model.state_dict(), "name": best.name,
                    "family": family, "metrics": best.metrics,
                    "num_classes": NUM_CLASSES, "class_remap": CLASS_REMAP,
                    "config": CONFIG["train"]}, path)
        saved.append(path)
fam_ml = [r for r in SEG_RESULTS.values()
          if r.family == "ClassicalML" and r.model is not None]
if fam_ml:
    best = max(fam_ml, key=lambda r: r.metrics["miou"])
    path = os.path.join(CONFIG["checkpoint_dir"], f"best_seg_ClassicalML_{best.name}.joblib")
    joblib.dump({"model": best.model, "scaler": SCALER, "name": best.name,
                 "feature_names": FEATURE_NAMES, "metrics": best.metrics}, path)
    saved.append(path)
if BEST_VOL_NAME and BEST_VOL_NAME in VOL_REG_MODELS:
    est, sc = VOL_REG_MODELS[BEST_VOL_NAME]
    path = os.path.join(CONFIG["checkpoint_dir"], f"best_volume_{BEST_VOL_NAME}")
    if isinstance(est, nn.Module):
        torch.save({"model_state": est.state_dict(), "name": BEST_VOL_NAME}, path + ".pth")
        saved.append(path + ".pth")
    else:
        joblib.dump({"model": est, "scaler": sc, "name": BEST_VOL_NAME,
                     "feature_names": VOL_FEATURE_NAMES,
                     "log_target": CONFIG["vol"]["log_target"]}, path + ".joblib")
        saved.append(path + ".joblib")
elif BEST_VOL_NAME:
    print(f"(best volume method '{BEST_VOL_NAME}' is a pure geometry formula — "
          "nothing to save)")
print("Saved best models only:")
for p in saved:
    print("  ", p)

# %% [markdown]
# ## 10 · Final Inference on `data/test` + Interactive Open3D Visualization
# 
# For every file in `data/test` (no labels needed): segment with the champion model,
# estimate volume with the best volume method, and open interactive Open3D windows —
# original cloud, predicted segmentation (auto palette, one distinct color per class),
# GT overlay when labels exist, plus bounding box / convex hull / voxel grid with the
# estimated volume in the window title. Close each window (`q`) to continue.
# If windows ever crash your kernel, set `CONFIG["vis"]["show_windows"]=False`.

# %%
# ================================================================ OPEN3D VISUALIZATION
def _o3d_show(geoms, title):
    if not CONFIG["vis"]["show_windows"]:
        log.info("[VIS] window skipped (show_windows=False): %s", title); return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            or sys.platform in ("win32", "darwin")):
        log.warning("[VIS] no display -> window skipped: %s", title); return
    try:
        w, h = CONFIG["vis"]["window"]
        o3d.visualization.draw_geometries(geoms, window_name=title, width=w, height=h)
    except Exception as e:
        log.warning("[VIS] Open3D window failed (%s)", e)

def colorize(pts, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64)-np.mean(pts.astype(np.float64),axis=0))
    pcd.colors = o3d.utility.Vector3dVector(PALETTE[np.clip(labels, 0,
                                                            NUM_CLASSES - 1)])
    return pcd

# def legend():
#     return " | ".join(f"{CLASS_NAMES[c]}=rgb{tuple((PALETTE[c]*255).astype(int))}"
#                       for c in range(NUM_CLASSES))

def legend():
    return " | ".join(f"{CLASS_NAMES[c]}=rgb{tuple(int(v) for v in (PALETTE[c]*255))}"
                      for c in range(NUM_CLASSES))

def visualize_sample(pts_raw, pred, gt=None, vol_info="", name=""):
    """শুধু segmented wood-powder (target) part দেখায় — ORIGINAL / PREDICTED
    (full cloud) / GROUND TRUTH (full cloud) window বাদ দেওয়া হয়েছে। convex-hull /
    voxel overlay-ও বাদ (সেগুলো volume-এ extra অংশ যোগ করে দেখাতো)। শুধু target
    points + axis-aligned bounding box দেখানো হয়।"""
    if not HAS_OPEN3D:
        log.warning("open3d missing -> visualization skipped"); return
    obj = pts_raw[pred == GT_CLASS] if GT_CLASS is not None else pts_raw
    if len(obj) <= 10:
        log.warning("[VIS] too few target points to show: %s", name); return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        obj.astype(np.float64) - np.mean(obj.astype(np.float64), axis=0))
    pcd.paint_uniform_color(PALETTE[GT_CLASS % len(PALETTE)])
    _o3d_show([pcd, pcd.get_axis_aligned_bounding_box()],
              f"TARGET (wood powder) | {name} | {vol_info}")
    if CONFIG["vis"]["save_ply"]:
        o3d.io.write_point_cloud(os.path.join(CONFIG["checkpoint_dir"],
                                              f"{name}_target.ply"),
                                 pcd)

# %%
# ================================================================ FINAL INFERENCE
def best_volume_estimate(pts):
    """Route to the winning volume method (learned model or geometry formula)."""
    if BEST_VOL_NAME is None:
        return float("nan"), "none"
    if BEST_VOL_NAME in GEO_VOLUME_FNS:
        return GEO_VOLUME_FNS[BEST_VOL_NAME](pts), BEST_VOL_NAME
    est, sc = VOL_REG_MODELS[BEST_VOL_NAME]
    if isinstance(est, nn.Module) and sc is None:
        return dl_estimate(est, pts), BEST_VOL_NAME
    if isinstance(est, nn.Module):                       # FeatMLP
        with torch.no_grad():
            x = torch.tensor(sc.transform(volume_features(pts).reshape(1, -1)),
                             dtype=torch.float32, device=HW.device)
            return max(float(np.expm1(est(x).item())), 0.0), BEST_VOL_NAME
    x = sc.transform(volume_features(pts).reshape(1, -1))
    v = float(est.predict(x)[0])
    return max(float(np.expm1(v)) if CONFIG["vol"]["log_target"] else v, 0.0), \
        BEST_VOL_NAME

infer_files = TEST_FILES or SPLIT["eval"]      # falls back to eval split if data/test empty
if not TEST_FILES:
    log.warning("data/test is empty -> demonstrating inference on the eval split")

INFER_ROWS = []
for f in infer_files:
    name = os.path.splitext(os.path.basename(f))[0]
    pts_raw, lbl_raw = load_pointcloud(f)
    gt = sanitize_labels(lbl_raw)
    entry = CACHE.get(f)
    TRACKER.start(f"inference_{name}", nested=True, tags={"phase": "inference"})
    t0 = time.perf_counter()
    # pred = (predict_full_cloud(BEST_SEG.model, entry)
    #         if BEST_SEG.model is not None else BEST_SEG.predictor(entry))
    pred = (full_cloud_labels(entry) if BEST_SEG.model is not None
            else BEST_SEG.predictor(entry))
    obj = pts_raw[pred == GT_CLASS] if GT_CLASS is not None else pts_raw
    vol, method = best_volume_estimate(obj)
    dt = time.perf_counter() - t0
    row = {"file": name, "n_points": len(pts_raw),
           "n_object_points": int(len(obj)),
           "estimated_volume": vol, "volume_method": method,
           "seg_model": BEST_SEG_NAME, "runtime_s": dt}
    if name in GT_VOLUMES:
        row["gt_volume"] = GT_VOLUMES[name]
        row["pct_error"] = abs(vol - GT_VOLUMES[name]) / max(
            abs(GT_VOLUMES[name]), 1e-9) * 100
    INFER_ROWS.append(row)
    TRACKER.metrics({"estimated_volume": vol, "runtime_s": dt,
                     "n_object_points": len(obj)})
    TRACKER.tag("seg_model", BEST_SEG_NAME); TRACKER.tag("volume_method", method)
    TRACKER.end()
    if len(INFER_ROWS) <= CONFIG["vis"]["max_vis_samples"]:
        visualize_sample(pts_raw, pred, gt,
                         vol_info=f"V={vol:.4f} ({method}) in {dt:.1f}s", name=name)

if isinstance(BEST_SEG.model, nn.Module):
    BEST_SEG.model.to("cpu")
HW.cleanup()                                      # all inference done -> VRAM free

INFER_TABLE = pd.DataFrame(INFER_ROWS)
print("FINAL INFERENCE RESULTS")
display(INFER_TABLE.round(4))

# %% [markdown]
# ### 10·A — Inference: per-file volume table with **all** algorithms (`data/test`)
# 
# For every inference file: segment once with the champion model, then run **every**
# volume method (geometry + all trained regressors) on the predicted object points.
# Files present in `gt_volume.csv` also get error columns.

# %%
# ================================================================ INFERENCE: ALL VOLUME METHODS PER FILE
def any_volume_estimate(name, pts):
    """Route any registered volume method (geometry / ML / DL / GNN) to its estimator."""
    try:
        if name in GEO_VOLUME_FNS:
            return float(GEO_VOLUME_FNS[name](pts))
        est, sc = VOL_REG_MODELS[name]
        if isinstance(est, nn.Module) and sc is None:          # point-based DL/GNN reg
            return float(dl_estimate(est, pts))
        if isinstance(est, nn.Module):                         # FeatMLP on features
            with torch.no_grad():
                x = torch.tensor(sc.transform(volume_features(pts).reshape(1, -1)),
                                 dtype=torch.float32, device=HW.device)
                return max(float(np.expm1(est.to(HW.device)(x).item())), 0.0)
        x = sc.transform(volume_features(pts).reshape(1, -1))  # sklearn regressor
        v = float(est.predict(x)[0])
        return max(float(np.expm1(v)) if CONFIG["vol"]["log_target"] else v, 0.0)
    except Exception as e:
        log.warning("[%s] estimate failed: %s", name, e)
        return float("nan")

_all_methods = ([n for n in GEO_VOLUME_FNS if n in VOL_RESULTS or
                 CONFIG["enabled"].get(n, False)]
                + [n for n in VOL_REG_MODELS if n not in GEO_VOLUME_FNS])
_ifiles = TEST_FILES or SPLIT["eval"]
if not TEST_FILES:
    log.warning("data/test empty -> demonstrating on eval split")

irows, ierrs = [], []
for f in tqdm(_ifiles, desc="all-method inference"):
    name = os.path.splitext(os.path.basename(f))[0]
    entry = CACHE.get(f)
    pred = (full_cloud_labels(entry) if BEST_SEG.model is not None
            else BEST_SEG.predictor(entry))
    pts_raw, _ = load_pointcloud(f)
    obj = pts_raw[pred == GT_CLASS] if GT_CLASS is not None else pts_raw
    row = {"file": name, "object_points": int(len(obj))}
    erow = {"file": name}
    gtv = GT_VOLUMES.get(name)
    if gtv is not None:
        row["gt_volume"] = gtv
    for m in _all_methods:
        v = any_volume_estimate(m, obj)
        row[m] = round(v, 4) if np.isfinite(v) else np.nan
        if gtv is not None:
            erow[m] = (round(abs(v - gtv) / max(abs(gtv), 1e-9) * 100, 2)
                       if np.isfinite(v) else np.nan)
    irows.append(row)
    if gtv is not None:
        ierrs.append(erow)
HW.cleanup()

INFER_ALL_TABLE = pd.DataFrame(irows)
print("=" * 110)
print("INFERENCE — VOLUME ESTIMATES BY EVERY ALGORITHM (data/test)")
print("=" * 110)
display(INFER_ALL_TABLE)

if ierrs:
    INFER_ERR_TABLE = pd.DataFrame(ierrs)
    _mc = [c for c in INFER_ERR_TABLE.columns if c != "file"]
    INFER_ERR_TABLE["best_method"] = INFER_ERR_TABLE[_mc].astype(float).idxmin(axis=1)
    print("\nINFERENCE — PER-FILE |% ERROR| vs gt_volume.csv (files with GT only)")
    display(INFER_ERR_TABLE)
    _sum = (INFER_ERR_TABLE[_mc].mean().sort_values().round(2)
            .rename("mean_pct_err").to_frame())
    print("\nMEAN |% ERROR| ACROSS INFERENCE FILES (lower = better)")
    display(_sum)
    plt.figure(figsize=(7, 0.32 * len(_sum) + 1.2))
    plt.barh(_sum.index, _sum["mean_pct_err"],
             color=["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(_sum))])
    plt.gca().invert_yaxis(); plt.xlabel("mean |% error|")
    plt.title("Inference: mean volume error by algorithm")
    plt.tight_layout(); plt.show()
else:
    log.info("No inference file matched gt_volume.csv -> estimates only (no error table)")

# %% [markdown]
# ---
# ## Conclusions & Learning Notes
# 
# - **Reading the benchmark:** mIoU/Dice rank segmentation quality; the composite
#   ranking folds in speed and memory — change `CONFIG["rank_weights"]` to match your
#   deployment constraints and rerun Section 8 only.
# - **Volume:** compare the `MAE` vs `MAE_gtseg` columns — a big gap means your first
#   investment should be a better segmenter, not a better volume formula. Geometry
#   methods fail systematically on non-watertight scans (see Section 6 intro), which is
#   why calibrated regression usually wins.
# - **Fairness caveats:** SVM/KNN were trained on capped subsamples (documented);
#   geometry segmenters are unsupervised and Hungarian-matched — their scores are
#   optimistic upper bounds for such methods.
# - **Scaling up:** set `quick_mode=False`, raise epochs, and rerun. Only best models
#   are saved; every run is fully tracked in MLflow at the configured URI.
# 

# %% [markdown]
# ## 19 · TIN-Based Volumetric Integration (Primary Method)
# 
# **Algorithm:**
# 1. SOR (Statistical Outlier Removal) — noise বাদ
# 2. DBSCAN → largest connected cluster → stray points বাদ
# 3. Ground plane fitting (RANSAC, perpendicular-height)
# 4. 2D Delaunay triangulation on plane-projected points
# 5. Alpha-shape boundary clipping (বড় boundary triangle বাদ)
# 6. TIN integration: Σ Area₂D(triangle) × mean_perpendicular_height
# 7. Bootstrap confidence interval (80 resamples)
# 
# **Output:** estimated volume (m³) + confidence interval

# %%
# ================================================================ SPEED FIX: keep model resident
import contextlib, types

@contextlib.contextmanager
def keep_model_resident(model=None):
    """Loop চলাকালীন model GPU-তে ধরে রাখে; per-file .to("cpu")/empty_cache() বন্ধ রাখে।
    with-block শেষে HW.cleanup / full_cloud_labels আগের আচরণে restore হয়।

    ব্যবহার:
        with keep_model_resident():
            ...  # Section 12 বা VOL_EVAL build বা inference loop এখানে
    """
    m = model if model is not None else getattr(BEST_SEG, "model", None)

    # 1) model একবার GPU-তে (torch হলে)
    if isinstance(m, nn.Module):
        m.to(HW.device); m.eval()

    # 2) HW.cleanup() no-op করে দিই (loop-এ per-file empty_cache/sync এড়াতে)
    _orig_cleanup = HW.cleanup
    HW.cleanup = types.MethodType(lambda self: None, HW)

    # 3) full_cloud_labels-এর ভেতরের per-cloud offload বাদ:
    #    torch হলে predict_full_cloud সরাসরি ব্যবহার করি, .to("cpu") ছাড়া।
    _orig_fcl = globals().get("full_cloud_labels")
    def _fast_full_cloud_labels(entry):
        if isinstance(BEST_SEG.model, nn.Module):
            return predict_full_cloud(BEST_SEG.model, entry)   # NO offload
        # classical ML path — অপরিবর্তিত
        n = len(entry["norm"]); out = np.empty(n, np.int64)
        for s in range(0, n, 100_000):
            idx = np.arange(s, min(s + 100_000, n))
            out[idx] = BEST_SEG.model.predict(
                SCALER.transform(point_features(entry["norm"], idx)))
        return out
    globals()["full_cloud_labels"] = _fast_full_cloud_labels

    try:
        yield
    finally:
        HW.cleanup = _orig_cleanup
        if _orig_fcl is not None:
            globals()["full_cloud_labels"] = _orig_fcl
        if isinstance(m, nn.Module):
            m.to("cpu")
        _orig_cleanup()          # block শেষে একবারই VRAM ছাড়ে

log.info("[SPEED] keep_model_resident() ready — Section 12 কে "
         "`with keep_model_resident():` দিয়ে wrap করুন দ্রুততার জন্য।")


# %%
# ================================================================ TIN VOLUME (PRIMARY METHOD)
import numpy as np
from scipy.spatial import Delaunay, cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# 1) Statistical Outlier Removal (SOR)
# ──────────────────────────────────────────────────────────────────────────────
def sor_filter(pts, k=16, std_ratio=2.0):
    """Remove points whose mean kNN distance > global_mean + std_ratio*std.
    Returns (filtered_pts, boolean_mask).  No geometry assumed — pure distance stats."""
    pts = np.asarray(pts, np.float64)
    if len(pts) <= k + 1:
        return pts, np.ones(len(pts), bool)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)           # col-0 = self (dist 0) → skip
    mean_dist = d[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    mask = mean_dist <= threshold
    return pts[mask], mask


# ──────────────────────────────────────────────────────────────────────────────
# 2) Largest DBSCAN cluster — stray / mis-labeled points বাদ
# ──────────────────────────────────────────────────────────────────────────────
def extract_main_cluster(pts, eps=0.15, min_samples=8):
    """Keep only the largest connected cluster.  Both Open3D and sklearn paths."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < min_samples * 2:
        return pts
    try:
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
        else:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        valid = labels[labels >= 0]
        if valid.size == 0:
            return pts
        main_lbl = np.bincount(valid).argmax()
        result = pts[labels == main_lbl]
        return result if len(result) >= min_samples else pts
    except Exception as e:
        log.warning("[TIN] cluster extraction failed (%s) -> using all points", e)
        return pts


# ──────────────────────────────────────────────────────────────────────────────
# 3) Robust ground-plane fitting (RANSAC on lower floor_pct% of points)
# ──────────────────────────────────────────────────────────────────────────────
def fit_ground_plane(pts, n_iter=500, thr=0.03, floor_pct=20):
    """Fit plane ax+by+cz+d=0 via RANSAC using only the lowest floor_pct% of points.
    Returns (unit_normal [3], d) where normal points upward (+z preferred)."""
    pts = np.asarray(pts, np.float64)
    z_thresh = np.percentile(pts[:, 2], floor_pct)
    cands = pts[pts[:, 2] <= z_thresh]
    if len(cands) < 10:
        cands = pts

    best_n   = np.array([0., 0., 1.])
    best_d   = -float(pts[:, 2].min())
    best_cnt = 0
    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        idx = rng.choice(len(cands), 3, replace=False)
        p0, p1, p2 = cands[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nm = np.linalg.norm(n)
        if nm < 1e-9:
            continue
        n /= nm
        if n[2] < 0:
            n = -n                          # keep upward-pointing
        d = -float(n @ p0)
        cnt = int((np.abs(pts @ n + d) < thr).sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, n.copy(), d

    return best_n, best_d


def perpendicular_heights(pts, normal, d):
    """Signed perpendicular distances: positive = above the plane."""
    return (pts @ normal + d).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Project 3D → 2D plane coordinates
# ──────────────────────────────────────────────────────────────────────────────
def project_to_plane(pts, normal):
    """Orthonormal 2D basis in the plane perpendicular to `normal`."""
    ref = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(normal, ref);  u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.column_stack([pts @ u, pts @ v])


# ──────────────────────────────────────────────────────────────────────────────
# 5) Alpha-shape clipping — বড় boundary triangle বাদ
# ──────────────────────────────────────────────────────────────────────────────
def alpha_clip_triangles(pts2d, triangles, alpha_mult=2.0):
    """Reject triangles whose longest edge > alpha_mult × median edge length.
    This removes the 'stretched' triangles at the cloud boundary that would
    otherwise inflate the volume estimate."""
    p0, p1, p2 = pts2d[triangles[:,0]], pts2d[triangles[:,1]], pts2d[triangles[:,2]]
    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e20 = np.linalg.norm(p0 - p2, axis=1)
    all_edges = np.concatenate([e01, e12, e20])
    valid_edges = all_edges[all_edges > 1e-9]
    if len(valid_edges) == 0:
        return triangles
    alpha = alpha_mult * np.median(valid_edges)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    return triangles[max_edge <= alpha]


# ──────────────────────────────────────────────────────────────────────────────
# 6) TIN integration — Σ Area2D(tri) × mean_height(tri)
# ──────────────────────────────────────────────────────────────────────────────
def tin_integrate(pts2d, heights, triangles):
    """Core volume computation via surface integral approximation."""
    p0, p1, p2 = pts2d[triangles[:,0]], pts2d[triangles[:,1]], pts2d[triangles[:,2]]
    h0, h1, h2 = heights[triangles[:,0]], heights[triangles[:,1]], heights[triangles[:,2]]
    cross_z = ((p1[:,0]-p0[:,0]) * (p2[:,1]-p0[:,1])
              - (p1[:,1]-p0[:,1]) * (p2[:,0]-p0[:,0]))
    area2d = np.abs(cross_z) / 2.0
    mean_h = (h0 + h1 + h2) / 3.0
    valid  = mean_h > 0                    # only integrate above-ground triangles
    return float(np.sum(area2d[valid] * mean_h[valid]))


# ──────────────────────────────────────────────────────────────────────────────
# 7) Bootstrap confidence interval
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_tin_confidence(pts_above, pts2d_above, heights_above, normal,
                              alpha_mult=2.0, n_bootstrap=80, subsample_frac=0.85,
                              seed=0):
    """Re-run TIN integration on random subsamples to estimate uncertainty."""
    rng = np.random.RandomState(seed)
    n = len(pts_above)
    n_sub = max(10, int(n * subsample_frac))
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n_sub, replace=False)
        sub2d = pts2d_above[idx]
        sub_h = heights_above[idx]
        try:
            tri = Delaunay(sub2d)
            tris = alpha_clip_triangles(sub2d, tri.simplices, alpha_mult)
            if len(tris) == 0:
                continue
            v = tin_integrate(sub2d, sub_h, tris)
            if np.isfinite(v) and v > 0:
                estimates.append(v)
        except Exception:
            pass
    if len(estimates) < 5:
        return None
    arr = np.array(estimates)
    return {"std": round(float(arr.std()), 5),
            "ci_low":  round(float(np.percentile(arr, 2.5)), 5),
            "ci_high": round(float(np.percentile(arr, 97.5)), 5)}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: vol_TIN — complete pipeline
# ──────────────────────────────────────────────────────────────────────────────
def vol_TIN(pts,
            sor_k=16, sor_std=2.0,
            cluster_eps=0.15,
            ransac_iter=500, ransac_thr=0.03, floor_pct=20,
            alpha_mult=2.0,
            run_bootstrap=True, n_bootstrap=80):
    """TIN-based volumetric integration:
      SOR → largest cluster → ground plane → heights → Delaunay →
      alpha-clip → TIN integral [→ bootstrap CI].
    Returns (volume_float, confidence_dict_or_None)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 10:
        return float("nan"), None

    # 1. SOR
    pts, _ = sor_filter(pts, k=sor_k, std_ratio=sor_std)
    if len(pts) < 10:
        return float("nan"), None

    # 2. Main cluster
    pts = extract_main_cluster(pts, eps=cluster_eps)
    if len(pts) < 10:
        return float("nan"), None

    # 3. Ground plane
    normal, d = fit_ground_plane(pts, n_iter=ransac_iter,
                                  thr=ransac_thr, floor_pct=floor_pct)

    # 4. Heights above plane
    heights = perpendicular_heights(pts, normal, d)
    above   = heights > 0
    if above.sum() < 10:
        heights = -heights           # plane may be inverted
        above   = heights > 0
    if above.sum() < 10:
        return float("nan"), None

    pts_ab = pts[above]
    h_ab   = heights[above]

    # 5. Project to 2D
    pts2d = project_to_plane(pts_ab, normal)

    # 6. Delaunay + alpha-clip
    if len(pts2d) < 4:
        return float("nan"), None
    try:
        tri      = Delaunay(pts2d)
        triangles = alpha_clip_triangles(pts2d, tri.simplices, alpha_mult)
    except Exception as e:
        log.warning("[TIN] Delaunay failed (%s)", e)
        return float("nan"), None
    if len(triangles) == 0:
        return float("nan"), None

    # 7. Integrate
    volume = tin_integrate(pts2d, h_ab, triangles)

    # 8. Bootstrap
    conf = None
    if run_bootstrap and len(pts_ab) >= 20:
        conf = bootstrap_tin_confidence(pts_ab, pts2d, h_ab, normal,
                                        alpha_mult=alpha_mult,
                                        n_bootstrap=n_bootstrap)
    return float(volume), conf


# Register for benchmark pipeline (no bootstrap for speed during benchmark)
GEO_VOLUME_FNS["vol_TIN"] = lambda pts: vol_TIN(pts, run_bootstrap=False)[0]
log.info("[TIN] vol_TIN registered in GEO_VOLUME_FNS — ready for benchmark + inference")


# %% [markdown]
# ### 19·A — TIN Volume: Final Inference Runner
# 
# Segment → SOR+cluster clean → TIN volume + bootstrap confidence → O3D visualization.
# 
# Console output per file:
# - Total Points / Target Points / Other Points / Target Ratio %
# - Estimated Volume (m³) + 95% confidence interval
# - Error vs GT (if available)
# 

# %%
# ================================================================ TIN INFERENCE RUNNER
assert GT_CLASS is not None,   "GT_CLASS undefined — Section 2 আগে চালান।"
assert "vol_TIN" in GEO_VOLUME_FNS, "Section 19 আগে চালান।"

_infer_files = TEST_FILES or SPLIT["eval"]
if not TEST_FILES:
    log.warning("data/test empty -> demo on eval split")

TIN_ROWS = []
with keep_model_resident():
    for _i, f in enumerate(_infer_files):
        name = os.path.splitext(os.path.basename(f))[0]
        pts_raw, _ = load_pointcloud(f)
        entry      = CACHE.get(f)

        # ── segmentation ──
        pred = (full_cloud_labels(entry) if BEST_SEG.model is not None
                else BEST_SEG.predictor(entry))

        n_total  = len(pts_raw)
        n_target = int((pred == GT_CLASS).sum())
        n_other  = n_total - n_target
        ratio    = n_target / max(n_total, 1) * 100

        # ── TIN volume with bootstrap confidence ──
        obj_raw       = pts_raw[pred == GT_CLASS]
        volume, conf  = vol_TIN(obj_raw)

        # ── console statistics ──
        print(f"\n{'─'*58}")
        print(f"  File          : {name}")
        print(f"  Total Points  : {n_total:,}")
        print(f"  Target Points : {n_target:,}")
        print(f"  Other Points  : {n_other:,}")
        print(f"  Target Ratio  : {ratio:.2f}%")
        print(f"  Est. Volume   : {volume:.4f} m³")
        if conf:
            print(f"  95% CI        : [{conf['ci_low']:.4f} – {conf['ci_high']:.4f}] m³"
                  f"  (±{conf['std']:.4f})")
        if name in GT_VOLUMES:
            gt  = GT_VOLUMES[name]
            pct = abs(volume - gt) / max(abs(gt), 1e-9) * 100
            print(f"  GT Volume     : {gt:.4f} m³  →  error {pct:.2f}%")
        print(f"{'─'*58}")

        # ── O3D visualization (target-only, after SOR+cluster) ──
        if _i < CONFIG["vis"]["max_vis_samples"] and HAS_OPEN3D and len(obj_raw) >= 10:
            pts_vis, _ = sor_filter(obj_raw)
            pts_vis    = extract_main_cluster(pts_vis)
            pcd = o3d.geometry.PointCloud()
            ctr = pts_vis.mean(axis=0)
            pcd.points = o3d.utility.Vector3dVector(
                (pts_vis - ctr).astype(np.float64))
            pcd.paint_uniform_color([0.0, 0.80, 0.0])        # green = target
            ci_str = (f" CI:[{conf['ci_low']:.3f}–{conf['ci_high']:.3f}]"
                      if conf else "")
            title = (f"TARGET | {name} | "
                     f"N_total={n_total:,}  N_target={n_target:,} ({ratio:.1f}%) | "
                     f"V={volume:.4f}m³{ci_str}")
            _o3d_show([pcd, pcd.get_axis_aligned_bounding_box()], title)

        # ── collect row ──
        row = {"file":             name,
               "total_points":     n_total,
               "target_points":    n_target,
               "other_points":     n_other,
               "target_ratio_pct": round(ratio, 2),
               "volume_m3":        round(volume, 5) if np.isfinite(volume) else None,
               "conf_std":    round(conf["std"],    5) if conf else None,
               "conf_ci_low": round(conf["ci_low"], 5) if conf else None,
               "conf_ci_high":round(conf["ci_high"],5) if conf else None,
               "seg_model":        BEST_SEG_NAME}
        if name in GT_VOLUMES:
            gt = GT_VOLUMES[name]
            row["gt_volume"] = gt
            row["pct_error"] = round(abs(volume - gt) / max(abs(gt), 1e-9) * 100, 2) \
                if np.isfinite(volume) else None
        TIN_ROWS.append(row)

if isinstance(BEST_SEG.model, nn.Module):
    BEST_SEG.model.to("cpu")
HW.cleanup()

TIN_TABLE = pd.DataFrame(TIN_ROWS)
print("\n" + "=" * 65)
print("TIN VOLUME RESULTS SUMMARY")
print("=" * 65)
display(TIN_TABLE.round(5))
if "pct_error" in TIN_TABLE.columns:
    print(f"\nMean % error across files: {TIN_TABLE['pct_error'].mean():.2f}%")



