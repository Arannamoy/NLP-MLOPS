# python 021_inferenct_gradio.py   →  opens http://127.0.0.1:7860
# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI for point-cloud segmentation.
# Contains NO inference logic of its own — everything is imported from
# inference.py, the shared, verified multi-architecture code path.
# Gradio hands uploads over as real file paths, so the standalone
# path-based pipeline is used directly, byte-for-byte.
#
# SYNCED with the multi-architecture inference.py (supports ptv2, kpconv,
# randlanet, spt, pointnet2, pointnext — pick the architecture in the UI
# that matches the checkpoint path you provide).
#
# Requirement: inference.py in the same folder.
# ─────────────────────────────────────────────────────────────────────────────
import os, time, hashlib, datetime
import numpy as np
import gradio as gr
import plotly.graph_objects as go

import multiprocessing as _mp

from inference import (
    TARGET_CLASS,
    ARCH_BUILDERS,
    load_model,
    predict_full_cloud,
    visualize_segmentation,
)

DEFAULT_MODEL_PATH = ("checkpoints/PointCloudDataset_SlidingWindow_9_channel/"
                      "PointTransformerV2_best.pth")


# ── Open3D window in a SEPARATE PROCESS ──────────────────────────────────────
# Gradio callbacks run in worker threads; Open3D/GLFW requires a main thread,
# so calling draw_geometries directly from a callback hangs or crashes.
# A child process gets its own main thread → window opens instantly and the
# Plotly output renders in parallel, without blocking the UI.
def _open3d_window(pts, preds, title, labels):
    visualize_segmentation(pts, preds, title=title, labels=labels)

def show_open3d_async(pts, preds, title, labels):
    ctx = _mp.get_context("fork" if hasattr(_mp, "get_context") and
                          "fork" in _mp.get_all_start_methods() else "spawn")
    p = ctx.Process(target=_open3d_window, args=(pts, preds, title, labels), daemon=True)
    p.start()          # do NOT join — window lives independently of the UI


# ── Model cache keyed on (file-content hash, architecture) ───────────────────
_MODEL_CACHE = {"sha": None, "arch": None, "model": None, "n_pts": None,
               "in_ch": None, "nc": None, "flags": None}

def _file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def get_model(model_path, arch):
    if not os.path.exists(model_path):
        raise gr.Error(f"Model file not found: {model_path}")
    sha = _file_sha(model_path)
    if _MODEL_CACHE["sha"] != sha or _MODEL_CACHE["arch"] != arch:
        model, n_pts, in_ch, nc, flags = load_model(model_path, arch)
        _MODEL_CACHE.update(sha=sha, arch=arch, model=model, n_pts=n_pts,
                            in_ch=in_ch, nc=nc, flags=flags)
    return (_MODEL_CACHE["model"], _MODEL_CACHE["n_pts"], _MODEL_CACHE["in_ch"],
           _MODEL_CACHE["nc"], _MODEL_CACHE["flags"], sha)


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
def run_segmentation(las_file, model_path, arch, max_pts, pt_size, open3d_window):
    if las_file is None:
        raise gr.Error("Upload a .las / .laz file first.")
    las_path = las_file if isinstance(las_file, str) else las_file.name

    model, n_pts, in_ch, nc, flags, model_sha = get_model(model_path, arch)
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
    fingerprint = (
        f"🔑 **Model** `{os.path.abspath(model_path)}` | arch `{arch}` | sha256 "
        f"`{model_sha[:12]}` | modified {mt:%Y-%m-%d %H:%M} | "
        f"in_channels={in_ch} | chunk size {n_pts:,}\n\n"
        f"📄 **File** `{os.path.basename(las_path)}` | sha256 "
        f"`{_file_sha(las_path)[:12]}`\n\n"
        f"Channel flags: `{flags}`"
    )

    t0 = time.time()
    pts, preds, lbl = predict_full_cloud(model, las_path, n_pts, in_ch, flags)
    dt = time.time() - t0

    n_tot = len(pts)
    n_tgt = int((preds == TARGET_CLASS).sum())
    stats = (
        f"| Total | 🟢 Target | 🔴 Others | Ratio | Time |\n"
        f"|---|---|---|---|---|\n"
        f"| {n_tot:,} | {n_tgt:,} | {n_tot - n_tgt:,} | "
        f"{n_tgt / max(n_tot, 1) * 100:.2f}% | {dt:.1f}s |"
    )

    # target-class IoU/Dice/Precision/Recall when the file carries GT labels
    has_labels = lbl.any() or (lbl == 0).sum() < len(lbl)
    if has_labels:
        tp = int(((lbl == TARGET_CLASS) & (preds == TARGET_CLASS)).sum())
        fp = int(((lbl != TARGET_CLASS) & (preds == TARGET_CLASS)).sum())
        fn = int(((lbl == TARGET_CLASS) & (preds != TARGET_CLASS)).sum())
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        stats += (f"\n\n📏 GT labels found → **target-class IoU = {iou:.4f}**  |  "
                 f"Dice = {dice:.4f}  |  Precision = {prec:.4f}  |  Recall = {rec:.4f}")

    if open3d_window:
        show_open3d_async(pts, preds,
                          title=f"Segmentation | {arch} | {os.path.basename(las_path)}",
                          labels=lbl if has_labels else None)
        stats += "\n\n🪟 Open3D window opened on your desktop (separate process)."

    fig = make_figure(pts, preds, max_pts, pt_size)
    return fingerprint, stats, fig


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Point Cloud Segmentation") as demo:
    gr.Markdown("# 🌲 Point Cloud Segmentation — Inference\n"
                f"Engine: `inference.py` (shared code path) | "
                f"Target class: `{TARGET_CLASS}` | "
                f"Architectures: {', '.join(ARCH_BUILDERS.keys())}")

    with gr.Row():
        las_in = gr.File(label="Upload .las / .laz", file_types=[".las", ".laz"])
        with gr.Column():
            arch_dd = gr.Dropdown(choices=list(ARCH_BUILDERS.keys()),
                                  value=list(ARCH_BUILDERS.keys())[0],
                                  label="Architecture")
            model_path_tb = gr.Textbox(value=DEFAULT_MODEL_PATH,
                                       label="Checkpoint path")
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
                  inputs=[las_in, model_path_tb, arch_dd, max_pts, pt_size, o3d_chk],
                  outputs=[fp_out, stats_out, plot_out])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
