# streamlit run inference.py
# ─────────────────────────────────────────────────────────────────────────────
# This app contains NO inference logic of its own. It imports load_model()
# and predict_full_cloud() from inference.py — the script that is
# verified to work. The uploaded file is written to disk and passed through
# the EXACT same code path. Divergence between app and script is therefore
# structurally impossible: same functions, same file, same model.
#
# Requirement: inference.py must sit in the same folder.
# ─────────────────────────────────────────────────────────────────────────────
import os, io, time, hashlib, tempfile, datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from inference import (
    MODEL_PATH, TARGET_CLASS, NUM_CLASSES,
    load_model as load_ptv2_model,
    predict_full_cloud,
)

st.set_page_config(page_title="Point Cloud Segmentation", layout="wide")
st.title("🌲 Point Cloud Segmentation — Inference")
st.caption(f"Model: `{MODEL_PATH}`  |  Target class: `{TARGET_CLASS}`  |  "
           f"Engine: `inference.py` (shared code path)")


# ── Model loading, cached on the FILE CONTENT hash ───────────────────────────
def _model_file_sha():
    if not os.path.exists(MODEL_PATH):
        return None
    return hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()

@st.cache_resource(show_spinner="Loading model…")
def cached_model(model_sha):
    """model_sha is part of the cache key → replacing best.pth on disk
    automatically invalidates this cache. No stale models, ever."""
    if model_sha is None:
        return None, None, f"Model file not found: {MODEL_PATH}"
    try:
        model, n_pts = load_ptv2_model(MODEL_PATH)   # standalone's loader
    except Exception as e:
        return None, None, f"Model load failed: {e}"
    return model, n_pts, None


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
uploaded = st.file_uploader("Upload a `.las` / `.laz` file", type=["las", "laz"])

c1, c2, c3 = st.columns(3)
max_pts  = c1.slider("Max display points", 20_000, 300_000, 100_000, 10_000)
pt_size  = c2.slider("Point size", 1, 6, 2)
vis_mode = c3.radio("Visualization", ["Open3D (window)", "Plotly (browser)"],
                    horizontal=True)

run_btn = st.button("▶  Run Segmentation", type="primary",
                    disabled=(uploaded is None))

if uploaded and run_btn:
    # 1) model — cached on content hash, loaded by the standalone loader
    model, n_pts, err = cached_model(_model_file_sha())
    if err:
        st.error(f"❌ {err}")
        st.stop()
    _mt = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    st.caption(f"🔑 Model `{os.path.abspath(MODEL_PATH)}` | "
               f"sha256 `{_model_file_sha()[:12]}` | modified {_mt:%Y-%m-%d %H:%M} | "
               f"chunk size {n_pts:,}")

    # 2) write the uploaded bytes to disk → the standalone path-based pipeline
    data = uploaded.getvalue()
    st.caption(f"📄 File `{uploaded.name}` | sha256 "
               f"`{hashlib.sha256(data).hexdigest()[:12]}` | "
               f"{len(data):,} bytes")
    suffix = os.path.splitext(uploaded.name)[1] or ".las"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # 3) inference — EXACTLY the standalone function
    try:
        t0 = time.time()
        with st.spinner("Running PTv2 inference (same code as "
                        "inference.py)…"):
            pts, preds, lbl = predict_full_cloud(model, tmp_path, n_pts)
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

    # per-file mIoU if the uploaded file carries ground-truth labels
    if np.unique(lbl).size > 1:
        ious = []
        for c in range(NUM_CLASSES):
            tp = int(((lbl == c) & (preds == c)).sum())
            fp = int(((lbl != c) & (preds == c)).sum())
            fn = int(((lbl == c) & (preds != c)).sum())
            if tp + fp + fn:
                ious.append(tp / (tp + fp + fn))
        st.info(f"📏 GT labels found in file → per-file mIoU = "
                f"{float(np.mean(ious)):.4f}")

    # 5) visualize
    st.subheader("3D Segmentation Result")
    if vis_mode == "Plotly (browser)":
        with st.spinner("Rendering…"):
            fig = make_figure(pts, preds, max_pts=max_pts, pt_size=pt_size)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing up to {max_pts:,} of {n_tot:,} points "
                   f"(inference ran on all {n_tot:,}).")
    else:
        from inference import visualize_segmentation
        st.info("🪟 Opening Open3D window on your desktop… "
                "Close the window to return here.")
        visualize_segmentation(pts, preds,
                               title=f"Segmentation | {uploaded.name}")
        st.success("✅ Open3D window closed.")

elif not uploaded:
    st.info("⬆️  Upload a `.las` file above, then press **Run Segmentation**.")