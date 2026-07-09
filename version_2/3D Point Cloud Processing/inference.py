# streamlit run app.py

import os, io, numpy as np, streamlit as st
import plotly.graph_objects as go
import laspy, joblib

MODEL_PATH   = "checkpoints/PointTransformerV2_epoch_0090.pth"
TARGET_CLASS = 1      
NUM_POINTS   = 4096   


EXT = os.path.splitext(MODEL_PATH)[1].lower()
if EXT in (".joblib", ".pkl"):
    MODEL_KIND = "sklearn"
elif EXT in (".pth", ".pt"):
    MODEL_KIND = "torch"
else:
    raise ValueError(f"Unknown model extension: {EXT}  (expected .joblib / .pth)")

st.set_page_config(page_title="Point Cloud Segmentation", layout="wide")
st.title("🌲 Point Cloud Segmentation — Inference")
st.caption(
    f"Model: `{MODEL_PATH}`  |  Type: **{MODEL_KIND}**  |  "
    f"Target class: `{TARGET_CLASS}`"
)


def compute_features(pts, k=16):
    """8-channel handcrafted features used during benchmark training."""
    from sklearn.neighbors import NearestNeighbors

    z     = pts[:, 2]
    z_rel = (z - z.min()) / max(z.max() - z.min(), 1e-9)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree",
                            n_jobs=-1).fit(pts)
    _, idx = nbrs.kneighbors(pts)          # (N, k)

    nb   = pts[idx]                        # (N, k, 3)
    mu   = nb.mean(1, keepdims=True)
    cov  = np.einsum("nki,nkj->nij", nb - mu, nb - mu) / k
    w, V = np.linalg.eigh(cov)            # ascending eigenvalues

    l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
    s          = np.clip(l1, 1e-12, None)
    linearity  = (l1 - l2) / s
    planarity  = (l2 - l3) / s
    sphericity =  l3 / s
    curvature  =  l3 / np.clip(l1 + l2 + l3, 1e-12, None)
    nz_abs     = np.abs(V[:, :, 0][:, 2])

    r_k     = np.linalg.norm(nb[:, -1, :] - pts, axis=1)
    density = k / np.clip((4 / 3) * np.pi * r_k ** 3, 1e-9, None)
    h_range = nb[:, :, 2].max(1) - nb[:, :, 2].min(1)

    X = np.column_stack([z_rel, nz_abs, linearity, planarity,
                         sphericity, curvature, np.log1p(density), h_range])
    return np.nan_to_num(X).astype(np.float32)

@st.cache_resource(show_spinner="Loading model…")
def load_model(_path=MODEL_PATH, _kind=MODEL_KIND):
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found:\n{MODEL_PATH}"

    # ── .joblib / .pkl  →  sklearn 
    if MODEL_KIND == "sklearn":
        blob = joblib.load(MODEL_PATH)
        if isinstance(blob, dict):
            clf    = blob.get("model",  blob)
            scaler = blob.get("scaler", None)
        else:
            clf, scaler = blob, None
        return ("sklearn", clf, scaler), None

    # ── .pth / .pt  →  PTv2 (torch) 
    import torch
    import torch.nn as nn

    try:
        from torch_cluster import knn as tc_knn
        from torch_scatter import scatter_softmax, scatter_add, \
                                   scatter_max,   scatter_mean
    except ImportError:
        return None, (
            "torch_cluster / torch_scatter not found.\n"
            "Install matching wheels from https://data.pyg.org/whl/"
        )

    ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    num_cls   = ckpt.get("num_classes", 2)
    IN        = 7   # 7-channel features used during training

    class GVA(nn.Module):
        """Grouped Vector Attention — core PTv2 attention layer."""
        def __init__(self, ch, g=6, k=16):
            super().__init__()
            assert ch % g == 0
            self.k, self.g, self.gc = k, g, ch // g
            self.q  = nn.Linear(ch, ch)
            self.kk = nn.Linear(ch, ch)
            self.v  = nn.Linear(ch, ch)
            self.pm = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
            self.pb = nn.Sequential(nn.Linear(3, ch), nn.ReLU(), nn.Linear(ch, ch))
            self.w  = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, g))

        def forward(self, x, pos, batch):
            e      = tc_knn(pos, pos, self.k, batch, batch)
            c, nb  = e[0], e[1]
            dp     = pos[c] - pos[nb]
            pb     = self.pb(dp)
            rel    = (self.q(x)[c] - self.kk(x)[nb]) * self.pm(dp) + pb
            wt     = scatter_softmax(self.w(rel), c, dim=0)
            vg     = (self.v(x)[nb] + pb).view(-1, self.g, self.gc)
            out    = scatter_add(vg * wt.unsqueeze(-1), c, dim=0,
                                 dim_size=x.size(0))
            return out.view(-1, self.g * self.gc)

    class Block(nn.Module):
        """Attention + MLP with LayerNorm and residual connections."""
        def __init__(self, ch, g=6, k=16):
            super().__init__()
            self.n1   = nn.LayerNorm(ch)
            self.n2   = nn.LayerNorm(ch)
            self.attn = GVA(ch, g, k)
            self.mlp  = nn.Sequential(
                nn.Linear(ch, ch * 2), nn.ReLU(), nn.Linear(ch * 2, ch))

        def forward(self, x, pos, batch):
            x = x + self.attn(self.n1(x), pos, batch)
            return x + self.mlp(self.n2(x))

    class GridPool(nn.Module):
        """Voxel grid pooling — downsamples the point cloud."""
        def __init__(self, in_ch, out_ch, grid):
            super().__init__()
            self.grid = grid
            self.proj = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU())

        def forward(self, x, pos, batch):
            vox      = torch.floor(pos / self.grid).long()
            key      = torch.cat([batch.unsqueeze(1), vox], dim=1)
            _, cl    = torch.unique(key, dim=0, return_inverse=True)
            xp, _    = scatter_max(self.proj(x), cl, dim=0)
            pos_pool = scatter_mean(pos, cl, dim=0)
            bat_pool = scatter_max(batch, cl, dim=0)[0]
            return xp, pos_pool, bat_pool, cl

    class PTv2Seg(nn.Module):
        """PointTransformerV2 encoder-decoder for per-point segmentation."""
        def __init__(self, nc, k=16, dims=(48, 96, 192), g=6):
            super().__init__()
            d = dims
            self.embed = nn.Sequential(
                nn.Linear(IN, d[0]), nn.ReLU(), nn.Linear(d[0], d[0]))
            self.e1 = Block(d[0], g, k);  self.p1 = GridPool(d[0], d[1], 0.08)
            self.e2 = Block(d[1], g, k);  self.p2 = GridPool(d[1], d[2], 0.16)
            self.e3 = Block(d[2], g, k)
            self.u2 = nn.Sequential(nn.Linear(d[2] + d[1], d[1]), nn.ReLU())
            self.d2 = Block(d[1], g, k)
            self.u1 = nn.Sequential(nn.Linear(d[1] + d[0], d[0]), nn.ReLU())
            self.d1 = Block(d[0], g, k)
            self.head = nn.Sequential(
                nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                nn.Dropout(0.4),    nn.Linear(128, nc))

        def forward(self, x):
            B, C, N = x.shape
            flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
            p0   = flat[:, :3].contiguous()
            b0   = torch.arange(B, device=x.device).repeat_interleave(N)

            h0             = self.e1(self.embed(flat), p0, b0)
            h1, p1, b1, c1 = self.p1(h0, p0, b0)
            h1             = self.e2(h1, p1, b1)
            h2, p2, b2, c2 = self.p2(h1, p1, b1)
            h2             = self.e3(h2, p2, b2)

            u1 = self.d2(self.u2(torch.cat([h1, h2[c2]], dim=1)), p1, b1)
            u0 = self.d1(self.u1(torch.cat([h0, u1[c1]], dim=1)), p0, b0)

            return self.head(u0).view(B, N, -1).permute(0, 2, 1)  # (B,nc,N)

    model = PTv2Seg(num_cls)
    state = ckpt.get("model_state", ckpt)
    try:
        model.load_state_dict(state)
    except Exception as e:
        return None, f"State dict mismatch: {e}"
    model.eval()
    return ("torch", model), None


def run_inference_sklearn(clf, scaler, pts):
    """Batched sklearn inference with progress bar."""
    n, batch = len(pts), 80_000
    preds    = np.empty(n, np.int64)
    bar      = st.progress(0, text="Computing features…")
    for s in range(0, n, batch):
        e    = min(s + batch, n)
        feat = compute_features(pts[s:e], k=16)
        if scaler is not None:
            feat = scaler.transform(feat)
        preds[s:e] = clf.predict(feat)
        bar.progress(e / n, text=f"Predicting… {e:,} / {n:,}")
    bar.empty()
    return preds


def run_inference_torch(model, pts):
    """Chunked full-cloud inference for PTv2."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7-channel features: norm-xyz + height + normals (zeros, no open3d needed)
    center = pts.mean(0, keepdims=True)
    scale  = max(np.linalg.norm(pts - center, axis=1).max(), 1e-9)
    nxyz   = ((pts - center) / scale).astype(np.float32)
    height = ((pts[:, 2] - pts[:, 2].min()) /
              max(pts[:, 2].max() - pts[:, 2].min(), 1e-6)).astype(np.float32)
    feat   = np.column_stack([nxyz, height[:, None],
                               np.zeros((len(pts), 3), np.float32)])

    n      = len(feat)
    perm   = np.random.RandomState(0).permutation(n)
    pad    = (NUM_POINTS - n % NUM_POINTS) % NUM_POINTS
    if pad:
        perm = np.concatenate([perm, perm[:pad]])
    chunks = perm.reshape(-1, NUM_POINTS)

    preds  = np.zeros(n, np.int64)
    bar    = st.progress(0, text="Running PTv2 inference…")
    with torch.no_grad():
        for i in range(0, len(chunks), 4):
            cid = chunks[i:i + 4]
            x   = torch.from_numpy(
                      feat[cid].transpose(0, 2, 1)).to(device)
            out = model(x).argmax(1).cpu().numpy()
            preds[cid.ravel()] = out.ravel()
            bar.progress(min((i + 4) / len(chunks), 1.0))
    bar.empty()
    model.to("cpu")
    return preds


def make_figure(pts, preds, max_pts, pt_size):
    n   = len(pts)
    idx = (np.random.RandomState(0).choice(n, max_pts, replace=False)
           if n > max_pts else np.arange(n))
    p, pr = pts[idx], preds[idx]

    tgt = pr == TARGET_CLASS
    oth = ~tgt

    traces = []

    # others — red
    if oth.any():
        q = p[oth]
        traces.append(go.Scatter3d(
            x=q[:, 0], y=q[:, 1], z=q[:, 2],
            mode="markers",
            marker=dict(size=pt_size, color="red", opacity=0.30),
            name=f"Others  ({oth.sum():,})"))

    # target — green
    if tgt.any():
        q = p[tgt]
        traces.append(go.Scatter3d(
            x=q[:, 0], y=q[:, 1], z=q[:, 2],
            mode="markers",
            marker=dict(size=pt_size, color="lime", opacity=0.80),
            name=f"Target  ({tgt.sum():,})"))

    # XYZ axes
    ctr  = pts.mean(0)
    span = (pts.max(0) - pts.min(0)).max() * 0.5
    for vec, col, lbl in [([1,0,0],"red","X"),
                           ([0,1,0],"lime","Y"),
                           ([0,0,1],"cyan","Z")]:
        end = ctr + np.array(vec, float) * span
        traces.append(go.Scatter3d(
            x=[ctr[0], end[0]], y=[ctr[1], end[1]], z=[ctr[2], end[2]],
            mode="lines+text",
            line=dict(color=col, width=5),
            text=["", lbl], textposition="top center",
            textfont=dict(color=col, size=14),
            showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(font=dict(size=13, color="white"),
                    bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.98),
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="#111",
                       gridcolor="#333", showbackground=True),
            yaxis=dict(title="Y", backgroundcolor="#111",
                       gridcolor="#333", showbackground=True),
            zaxis=dict(title="Z", backgroundcolor="#111",
                       gridcolor="#333", showbackground=True),
            bgcolor="#111",
            aspectmode="data"),
        paper_bgcolor="#0E1117",
        font=dict(color="white"))
    return fig



uploaded = st.file_uploader(
    "Upload a `.las` / `.laz` file", type=["las", "laz"])

c1, c2, c3 = st.columns(3)
max_pts = c1.slider("Max display points",
                    20_000, 300_000, 100_000, 10_000)
pt_size = c2.slider("Point size", 1, 6, 2)
vis_mode = c3.radio("Visualization", ["Plotly (browser)", "Open3D (window)"],
                    horizontal=True)

run_btn = st.button("▶  Run Segmentation", type="primary",
                    disabled=(uploaded is None))

if uploaded and run_btn:

    # 1. load model (cached)
    model_obj, err = load_model(MODEL_PATH, MODEL_KIND)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    # 2. read point cloud
    with st.spinner("Reading point cloud…"):
        las = laspy.read(io.BytesIO(uploaded.read()))
        pts = np.column_stack([np.asarray(las.x, np.float64),
                               np.asarray(las.y, np.float64),
                               np.asarray(las.z, np.float64)])
    st.success(f"✅ {len(pts):,} points loaded from `{uploaded.name}`")

    # 3. center the point cloud (remove GPS offset, same as O3D visualization)
    pts = pts - pts.mean(axis=0)

    # 3. inference — extension-driven branch
    if model_obj[0] == "sklearn":
        _, clf, scaler = model_obj
        preds = run_inference_sklearn(clf, scaler, pts)
    else:
        _, model = model_obj
        preds = run_inference_torch(model, pts)

    # 4. statistics
    n_tot = len(pts)
    n_tgt = int((preds == TARGET_CLASS).sum())
    n_oth = n_tot - n_tgt

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Points",  f"{n_tot:,}")
    m2.metric("🟢 Target",     f"{n_tgt:,}")
    m3.metric("🔴 Others",     f"{n_oth:,}")
    m4.metric("Target Ratio",  f"{n_tgt / max(n_tot,1) * 100:.2f}%")

    # 5. visualize — user choice
    st.subheader("3D Segmentation Result")

    if vis_mode == "Plotly (browser)":
        with st.spinner("Rendering…"):
            fig = make_figure(pts, preds, max_pts=max_pts, pt_size=pt_size)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Showing up to {max_pts:,} of {n_tot:,} points "
            f"(inference ran on all {n_tot:,}).")

    else:  # Open3D window
        try:
            import open3d as o3d
        except ImportError:
            st.error("❌ open3d not installed. Run:  pip install open3d")
            st.stop()

        st.info("🪟 Opening Open3D window on your desktop…  "
                "Close the window to return here.")

        # build coloured point cloud — green=target, red=others
        colors = np.zeros((len(pts), 3), dtype=np.float64)
        colors[preds == TARGET_CLASS] = [0.0, 0.75, 0.0]   # green
        colors[preds != TARGET_CLASS] = [0.75, 0.0, 0.0]   # red

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # XYZ axes
        span = float((pts.max(0) - pts.min(0)).max()) * 0.5
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=span, origin=[0.0, 0.0, 0.0])

        o3d.visualization.draw_geometries(
            [pcd, axes],
            window_name=(f"Segmentation | {uploaded.name} | "
                         f"target={n_tgt:,} ({n_tgt/max(n_tot,1)*100:.1f}%) | "
                         f"others={n_oth:,}"),
            width=1280, height=800)
        st.success("✅ Open3D window closed.")

elif not uploaded:
    st.info("⬆️  Upload a `.las` file above, then press **Run Segmentation**.")