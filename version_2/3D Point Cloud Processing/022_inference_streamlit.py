# streamlit run 022_inference_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# This app contains NO inference logic of its own. It imports load_model(),
# predict_full_cloud(), visualize_segmentation(), and ARCH_BUILDERS from
# inference.py — the shared, verified code path. The uploaded file is
# written to disk and passed through the EXACT same functions used by the
# CLI script. Divergence between app and script is structurally impossible.
#
# SYNCED with the multi-architecture inference.py (supports ptv2, kpconv,
# randlanet, spt, pointnet2, pointnext — pick the architecture in the UI
# that matches the checkpoint you upload/select).
#
# Requirement: inference.py must sit in the same folder.
# ─────────────────────────────────────────────────────────────────────────────
import os, time, hashlib, tempfile, datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from inference import (
    TARGET_CLASS,
    ARCH_BUILDERS,
    load_model,
    predict_full_cloud,
    visualize_segmentation,
)

st.set_page_config(page_title="Point Cloud Segmentation", layout="wide")
st.title("🌲 Point Cloud Segmentation — Inference")
st.caption("Engine: `inference.py` (shared code path) | "
           f"Target class: `{TARGET_CLASS}` | "
           f"Architectures: {', '.join(ARCH_BUILDERS.keys())}")


# ── Model loading, cached on (file content hash, architecture) ───────────────
def _model_file_sha(path):
    if not path or not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

@st.cache_resource(show_spinner="Loading model…")
def cached_model(model_sha, model_path, arch):
    """model_sha + arch are part of the cache key → replacing the checkpoint
    file on disk, OR switching architecture, automatically invalidates this
    cache. No stale models, ever."""
    if model_sha is None:
        return None, None, None, None, None, f"Model file not found: {model_path}"
    try:
        model, n_pts, in_ch, nc, flags = load_model(model_path, arch)
    except Exception as e:
        return None, None, None, None, None, f"Model load failed: {e}"
    return model, n_pts, in_ch, nc, flags, None


# ── Plotly visualization (display only — no effect on predictions) ───────────
def make_figure(pts, preds, max_pts, pt_size):
    n   = len(pts)
    idx = (np.random.RandomState(0).choice(n, max_pts, replace=False)
           if n > max_pts else np.arange(n))
    p, pr = pts[idx], preds[idx]
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


# ── UI ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    arch = st.selectbox("Architecture", list(ARCH_BUILDERS.keys()))
    model_path = st.text_input(
        "Checkpoint path",
        value="checkpoints/PointCloudDataset_SlidingWindow_9_channel/PointTransformerV2_best.pth")

uploaded = st.file_uploader("Upload a `.las` / `.laz` file", type=["las", "laz"])

c1, c2, c3 = st.columns(3)
max_pts  = c1.slider("Max display points", 20_000, 300_000, 100_000, 10_000)
pt_size  = c2.slider("Point size", 1, 6, 2)
vis_mode = c3.radio("Visualization", ["Open3D (window)", "Plotly (browser)"],
                    horizontal=True)

run_btn = st.button("▶  Run Segmentation", type="primary",
                    disabled=(uploaded is None))

if uploaded and run_btn:
    # 1) model — cached on (content hash, architecture)
    sha = _model_file_sha(model_path)
    model, n_pts, in_ch, nc, flags, err = cached_model(sha, model_path, arch)
    if err:
        st.error(f"❌ {err}")
        st.stop()
    _mt = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
    st.caption(f"🔑 Model `{os.path.abspath(model_path)}` | arch `{arch}` | "
               f"sha256 `{sha[:12]}` | modified {_mt:%Y-%m-%d %H:%M} | "
               f"in_channels={in_ch} | chunk size {n_pts:,}")
    st.caption(f"Channel flags: {flags}")

    # 2) write the uploaded bytes to disk → the standalone path-based pipeline
    data = uploaded.getvalue()
    st.caption(f"📄 File `{uploaded.name}` | sha256 "
               f"`{hashlib.sha256(data).hexdigest()[:12]}` | "
               f"{len(data):,} bytes")
    suffix = os.path.splitext(uploaded.name)[1] or ".las"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # 3) inference — EXACTLY the shared function, same args the CLI script uses
    try:
        t0 = time.time()
        with st.spinner(f"Running {arch} inference (same code as inference.py)…"):
            pts, preds, lbl = predict_full_cloud(model, tmp_path, n_pts, in_ch, flags)
        st.caption(f"⏱️ Inference finished in {time.time() - t0:.1f}s")
    except Exception as e:
        st.error("❌ Inference failed — full traceback below.")
        st.exception(e)
        st.stop()
    finally:
        os.unlink(tmp_path)

    # 4) statistics
    n_tot = len(pts)
    n_tgt = int((preds == TARGET_CLASS).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Points", f"{n_tot:,}")
    m2.metric("🟢 Target",    f"{n_tgt:,}")
    m3.metric("🔴 Others",    f"{n_tot - n_tgt:,}")
    m4.metric("Target Ratio", f"{n_tgt / max(n_tot, 1) * 100:.2f}%")

    # per-file target-class IoU/Dice/Precision/Recall if the file carries GT labels
    has_labels = lbl.any() or (lbl == 0).sum() < len(lbl)
    if has_labels:
        tp = int(((lbl == TARGET_CLASS) & (preds == TARGET_CLASS)).sum())
        fp = int(((lbl != TARGET_CLASS) & (preds == TARGET_CLASS)).sum())
        fn = int(((lbl == TARGET_CLASS) & (preds != TARGET_CLASS)).sum())
        tn = int(((lbl != TARGET_CLASS) & (preds != TARGET_CLASS)).sum())
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        st.info(f"📏 GT labels found → **target-class IoU = {iou:.4f}**  |  "
               f"Dice = {dice:.4f}  |  Precision = {prec:.4f}  |  Recall = {rec:.4f}")
        st.caption(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # 5) visualize
    st.subheader("3D Segmentation Result")
    if vis_mode == "Plotly (browser)":
        with st.spinner("Rendering…"):
            fig = make_figure(pts, preds, max_pts=max_pts, pt_size=pt_size)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing up to {max_pts:,} of {n_tot:,} points "
                   f"(inference ran on all {n_tot:,}).")
    else:
        st.info("🪟 Opening Open3D window on your desktop… "
                "Close the window to return here.")
        visualize_segmentation(pts, preds,
                               title=f"Segmentation | {arch} | {uploaded.name}",
                               labels=lbl if has_labels else None)
        st.success("✅ Open3D window closed.")

elif not uploaded:
    st.info("⬆️  Upload a `.las` file above, then press **Run Segmentation**.")
