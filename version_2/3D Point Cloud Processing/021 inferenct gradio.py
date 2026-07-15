# python inference_gradio.py   →  opens http://127.0.0.1:7860
# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI for PTv2 point-cloud segmentation.
# Contains NO inference logic of its own — everything is imported from
# inference_standalone.py, the verified single source of truth.
# Gradio hands uploads over as real file paths, so the standalone
# path-based pipeline is used directly, byte-for-byte.
#
# Requirement: inference_standalone.py in the same folder.
# ─────────────────────────────────────────────────────────────────────────────
import os, time, hashlib, datetime
import numpy as np
import gradio as gr
import plotly.graph_objects as go

import multiprocessing as _mp

from inference_standalone import (
    MODEL_PATH, TARGET_CLASS, NUM_CLASSES,
    load_model, predict_full_cloud,
)

# ── Open3D window in a SEPARATE PROCESS ──────────────────────────────────────
# Gradio callbacks run in worker threads; Open3D/GLFW requires a main thread,
# so calling draw_geometries directly from a callback hangs or crashes.
# A child process gets its own main thread → window opens instantly and the
# Plotly output renders in parallel, without blocking the UI.
def _open3d_window(pts, preds, title):
    import numpy as np, open3d as o3d
    colors = np.zeros((len(pts), 3), dtype=np.float64)
    colors[preds == TARGET_CLASS] = [0.0, 0.8, 0.0]
    colors[preds != TARGET_CLASS] = [0.8, 0.0, 0.0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((pts - pts.mean(axis=0)).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    span = float((pts.max(0) - pts.min(0)).max()) * 0.5
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=span)
    n_t = int((preds == TARGET_CLASS).sum())
    o3d.visualization.draw_geometries(
        [pcd, axes],
        window_name=f"{title} | green(target)={n_t:,} "
                    f"({n_t / max(len(preds), 1) * 100:.1f}%) | "
                    f"red(others)={len(preds) - n_t:,}",
        width=1280, height=800)

def show_open3d_async(pts, preds, title):
    ctx = _mp.get_context("fork" if hasattr(_mp, "get_context") and
                          "fork" in _mp.get_all_start_methods() else "spawn")
    p = ctx.Process(target=_open3d_window, args=(pts, preds, title), daemon=True)
    p.start()          # do NOT join — window lives independently of the UI

# ── Model cache keyed on file-content hash (auto-reload when best.pth changes) ──
_MODEL_CACHE = {"sha": None, "model": None, "n_pts": None}

def _file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise gr.Error(f"Model file not found: {MODEL_PATH}")
    sha = _file_sha(MODEL_PATH)
    if _MODEL_CACHE["sha"] != sha:                 # new/changed file → reload
        model, n_pts = load_model(MODEL_PATH)
        _MODEL_CACHE.update(sha=sha, model=model, n_pts=n_pts)
    return _MODEL_CACHE["model"], _MODEL_CACHE["n_pts"], sha


# ── Plotly 3D figure (display only) ──────────────────────────────────────────
def make_figure(pts, preds, max_pts, pt_size):
    n   = len(pts)
    idx = (np.random.RandomState(0).choice(n, int(max_pts), replace=False)
           if n > max_pts else np.arange(n))
    p, pr = pts[idx] - pts.mean(axis=0), preds[idx]      # centre for display
    tgt = pr == TARGET_CLASS

    traces = []
    if (~tgt).any():
        q = p[~tgt]
        traces.append(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                   marker=dict(size=pt_size, color="red", opacity=0.30),
                                   name=f"Others  ({int((~tgt).sum()):,})"))
    if tgt.any():
        q = p[tgt]
        traces.append(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                   marker=dict(size=pt_size, color="lime", opacity=0.80),
                                   name=f"Target  ({int(tgt.sum()):,})"))
    fig = go.Figure(traces)
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(font=dict(size=13, color="white"),
                    bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.98),
        scene=dict(xaxis=dict(title="X", backgroundcolor="#111", gridcolor="#333",
                              showbackground=True),
                   yaxis=dict(title="Y", backgroundcolor="#111", gridcolor="#333",
                              showbackground=True),
                   zaxis=dict(title="Z", backgroundcolor="#111", gridcolor="#333",
                              showbackground=True),
                   bgcolor="#111", aspectmode="data"),
        paper_bgcolor="#0E1117", font=dict(color="white"))
    return fig


# ── Main callback ─────────────────────────────────────────────────────────────
def run_segmentation(las_file, max_pts, pt_size, open3d_window):
    if las_file is None:
        raise gr.Error("Upload a .las / .laz file first.")
    las_path = las_file if isinstance(las_file, str) else las_file.name

    model, n_pts, model_sha = get_model()
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    fingerprint = (
        f"🔑 **Model** `{os.path.abspath(MODEL_PATH)}` | sha256 "
        f"`{model_sha[:12]}` | modified {mt:%Y-%m-%d %H:%M} | "
        f"chunk size {n_pts:,}\n\n"
        f"📄 **File** `{os.path.basename(las_path)}` | sha256 "
        f"`{_file_sha(las_path)[:12]}`"
    )

    t0 = time.time()
    pts, preds, lbl = predict_full_cloud(model, las_path, n_pts)   # shared code
    dt = time.time() - t0

    n_tot = len(pts)
    n_tgt = int((preds == TARGET_CLASS).sum())
    stats = (
        f"| Total | 🟢 Target | 🔴 Others | Ratio | Time |\n"
        f"|---|---|---|---|---|\n"
        f"| {n_tot:,} | {n_tgt:,} | {n_tot - n_tgt:,} | "
        f"{n_tgt / max(n_tot, 1) * 100:.2f}% | {dt:.1f}s |"
    )

    # exact per-file mIoU when the file carries GT labels
    if np.unique(lbl).size > 1:
        ious = []
        for c in range(NUM_CLASSES):
            tp = int(((lbl == c) & (preds == c)).sum())
            fp = int(((lbl != c) & (preds == c)).sum())
            fn = int(((lbl == c) & (preds != c)).sum())
            if tp + fp + fn:
                ious.append(tp / (tp + fp + fn))
        stats += f"\n\n📏 GT labels found → **per-file mIoU = {np.mean(ious):.4f}**"

    if open3d_window:
        show_open3d_async(pts, preds,
                          title=f"Segmentation | {os.path.basename(las_path)}")
        stats += "\n\n🪟 Open3D window opened on your desktop (separate process)."

    fig = make_figure(pts, preds, max_pts, pt_size)
    return fingerprint, stats, fig


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Point Cloud Segmentation") as demo:
    gr.Markdown("# 🌲 Point Cloud Segmentation — Inference\n"
                f"Model: `{MODEL_PATH}` | Target class: `{TARGET_CLASS}` | "
                "Engine: `inference_standalone.py` (shared code path)")

    with gr.Row():
        las_in = gr.File(label="Upload .las / .laz", file_types=[".las", ".laz"])
        with gr.Column():
            max_pts = gr.Slider(20_000, 300_000, value=100_000, step=10_000,
                                label="Max display points")
            pt_size = gr.Slider(1, 6, value=2, step=1, label="Point size")
            o3d_chk = gr.Checkbox(
                label="Also open Open3D desktop window (alongside Plotly)",
                value=False)
            run_btn = gr.Button("▶ Run Segmentation", variant="primary")

    fp_out    = gr.Markdown()
    stats_out = gr.Markdown()
    plot_out  = gr.Plot(label="3D Segmentation Result")

    run_btn.click(run_segmentation,
                  inputs=[las_in, max_pts, pt_size, o3d_chk],
                  outputs=[fp_out, stats_out, plot_out])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())