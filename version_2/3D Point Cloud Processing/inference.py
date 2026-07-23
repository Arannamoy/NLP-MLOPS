import sys, os, io, argparse
import numpy as np
import laspy
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

TARGET_CLASS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

PREP = {
    "voxel_size": 0.01, "normal_radius": 0.05, "normal_max_nn": 30,
    "sor_enabled": True, "sor_k": 16, "sor_std": 2.0,
    "cluster_filter_enabled": True, "cluster_eps": 0.03, "cluster_min_size": 100,
    "geom_radius": 0.06, "density_k": 16,
}

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
        e = tc_knn(pos, pos, min(self.k, x.size(0)), batch, batch)
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
    def __init__(self, nc, in_channels, dims=(48, 96, 192), g=6, k=16,
                grid_base=0.08, grid_mult=2.0):
        super().__init__()
        d = list(dims)
        n = len(d)
        assert n >= 2, "dims must have at least 2 stages"
        self.n_stages = n
        self.embed = nn.Sequential(
            nn.Linear(in_channels, d[0]), nn.ReLU(),
            nn.Linear(d[0], d[0]))
        self.enc_blocks = nn.ModuleList([Block(d[i], g, k) for i in range(n)])
        self.pools = nn.ModuleList([
            GridPool(d[i], d[i + 1], grid_base * (grid_mult ** i))
            for i in range(n - 1)])
        self.dec_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d[i + 1] + d[i], d[i]), nn.ReLU())
            for i in range(n - 1)])
        self.dec_blocks = nn.ModuleList([Block(d[i], g, k) for i in range(n - 1)])
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat  = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        pos   = flat[:, :3].contiguous()
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        h = self.enc_blocks[0](self.embed(flat), pos, batch)
        hs, poss, batches, cls = [h], [pos], [batch], []
        for i in range(self.n_stages - 1):
            h, pos, batch, cl = self.pools[i](h, poss[-1], batches[-1])
            cls.append(cl)
            h = self.enc_blocks[i + 1](h, pos, batch)
            hs.append(h); poss.append(pos); batches.append(batch)

        u = hs[-1]
        for i in reversed(range(self.n_stages - 1)):
            u = self.dec_proj[i](torch.cat([hs[i], u[cls[i]]], 1))
            u = self.dec_blocks[i](u, poss[i], batches[i])
        return self.head(u).view(B, N, -1).permute(0, 2, 1)


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
        denom = scatter_add(corr.sum(1, keepdim=True), c, dim=0, dim_size=x.size(0)).clamp(min=1e-6)
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
    def __init__(self, nc, in_channels, kp_kernel_points=15, kp_neighbors=16,
                kp_radius=0.06, kp_dims=(48, 96, 192), grid_base=0.08, grid_mult=2.0):
        super().__init__()
        d = list(kp_dims)
        n = len(d)
        assert n >= 2, "kp_dims must have at least 2 stages"
        self.n_stages = n
        r = kp_radius
        self.embed = nn.Sequential(
            nn.Linear(in_channels, d[0]), nn.ReLU(),
            nn.Linear(d[0], d[0]))
        self.enc_blocks = nn.ModuleList([
            KPBlock(d[i], r * (2 ** i), kp_kernel_points, kp_neighbors)
            for i in range(n)])
        self.pools = nn.ModuleList([
            GridPool(d[i], d[i + 1], grid_base * (grid_mult ** i))
            for i in range(n - 1)])
        self.dec_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d[i + 1] + d[i], d[i]), nn.ReLU())
            for i in range(n - 1)])
        self.dec_blocks = nn.ModuleList([
            KPBlock(d[i], r * (2 ** i), kp_kernel_points, kp_neighbors)
            for i in range(n - 1)])
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat  = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        pos   = flat[:, :3].contiguous()
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        h = self.enc_blocks[0](self.embed(flat), pos, batch)
        hs, poss, batches, cls = [h], [pos], [batch], []
        for i in range(self.n_stages - 1):
            h, pos, batch, cl = self.pools[i](h, poss[-1], batches[-1])
            cls.append(cl)
            h = self.enc_blocks[i + 1](h, pos, batch)
            hs.append(h); poss.append(pos); batches.append(batch)

        u = hs[-1]
        for i in reversed(range(self.n_stages - 1)):
            u = self.dec_proj[i](torch.cat([hs[i], u[cls[i]]], 1))
            u = self.dec_blocks[i](u, poss[i], batches[i])
        return self.head(u).view(B, N, -1).permute(0, 2, 1)


class LocSE(nn.Module):
    def __init__(self, d_half):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(10, d_half), nn.ReLU())

    def forward(self, x, pos, c, nb):
        rel  = pos[c] - pos[nb]
        dist = rel.norm(dim=1, keepdim=True)
        g = torch.cat([pos[c], pos[nb], rel, dist], 1)
        return torch.cat([self.mlp(g), x[nb]], 1)


class AttPool(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.score = nn.Linear(d_in, d_in, bias=False)
        self.mlp   = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())

    def forward(self, f, c, n_points):
        w   = scatter_softmax(self.score(f), c, dim=0)
        agg = scatter_add(f * w, c, dim=0, dim_size=n_points)
        return self.mlp(agg)


class DilatedResBlock(nn.Module):
    def __init__(self, d_in, d_out, neighbors):
        super().__init__()
        dh = d_out // 2
        self.pre   = nn.Sequential(nn.Linear(d_in, dh), nn.ReLU())
        self.se1   = LocSE(dh // 2)
        self.at1   = AttPool(dh // 2 + dh, dh)
        self.se2   = LocSE(dh // 2)
        self.at2   = AttPool(dh // 2 + dh, d_out)
        self.short = nn.Linear(d_in, d_out)
        self.k     = neighbors

    def forward(self, x, pos, batch):
        e = tc_knn(pos, pos, self.k, batch, batch)
        c, nb = e[0], e[1]
        h = self.pre(x)
        h = self.at1(self.se1(h, pos, c, nb), c, x.size(0))
        h = self.at2(self.se2(h, pos, c, nb), c, x.size(0))
        return torch.relu(h + self.short(x))


def _random_subsample(x, pos, batch, B, n_per, ratio):
    keep = max(n_per // ratio, 1)
    idx  = torch.cat([b * n_per + torch.randperm(n_per, device=x.device)[:keep] for b in range(B)])
    idx, _ = idx.sort()
    return x[idx], pos[idx], batch[idx], keep


class RandLANetSeg(nn.Module):
    def __init__(self, nc, in_channels, rl_neighbors=16, rl_decimation=4, rl_dims=(48, 96, 192)):
        super().__init__()
        d = rl_dims
        self.ratio = rl_decimation
        self.embed = nn.Sequential(nn.Linear(in_channels, d[0]), nn.ReLU(), nn.Linear(d[0], d[0]))
        self.e1 = DilatedResBlock(d[0], d[0], rl_neighbors)
        self.e2 = DilatedResBlock(d[0], d[1], rl_neighbors)
        self.e3 = DilatedResBlock(d[1], d[2], rl_neighbors)
        self.u2 = nn.Sequential(nn.Linear(d[2] + d[1], d[1]), nn.ReLU())
        self.u1 = nn.Sequential(nn.Linear(d[1] + d[0], d[0]), nn.ReLU())
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def _upsample(self, x_coarse, pos_coarse, b_coarse, pos_fine, b_fine):
        e = tc_knn(pos_coarse, pos_fine, 1, b_coarse, b_fine)
        return x_coarse[e[1]]

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        h0 = self.e1(self.embed(flat), p0, b0)
        h1, p1, b1, n1 = _random_subsample(h0, p0, b0, B, N, self.ratio)
        h1 = self.e2(h1, p1, b1)
        h2, p2, b2, n2 = _random_subsample(h1, p1, b1, B, n1, self.ratio)
        h2 = self.e3(h2, p2, b2)
        u1 = self.u2(torch.cat([h1, self._upsample(h2, p2, b2, p1, b1)], 1))
        u0 = self.u1(torch.cat([h0, self._upsample(u1, p1, b1, p0, b0)], 1))
        return self.head(u0).view(B, N, -1).permute(0, 2, 1)


class SPBlock(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.attn = GVA(ch, g=6, k=k)
        self.mlp  = nn.Sequential(nn.Linear(ch, ch * 2), nn.ReLU(), nn.Linear(ch * 2, ch))

    def forward(self, x, pos, batch):
        x = x + self.attn(self.n1(x), pos, batch)
        return x + self.mlp(self.n2(x))


def _scatter_min(src, idx, dim_size):
    buf = torch.full((dim_size,), -float("inf"), device=src.device, dtype=src.dtype)
    buf.index_reduce_(0, idx, -src, "amax", include_self=True)
    return -buf


def _geometric_partition(pos, normals, geom, batch, k, tau_normal, tau_geom, n_iters):
    n = pos.size(0)
    e = tc_knn(pos, pos, min(k, n), batch, batch)
    c, nb = e[0], e[1]
    cos = (normals[c] * normals[nb]).sum(1)
    keep = cos > tau_normal
    if geom is not None:
        geom_diff = (geom[c] - geom[nb]).abs().sum(1)
        keep = keep & (geom_diff < tau_geom)
    c, nb = c[keep], nb[keep]
    labels = torch.arange(n, device=pos.device, dtype=pos.dtype)
    for _ in range(n_iters):
        prop = _scatter_min(labels[nb], c, n)
        prop = torch.minimum(prop, labels)
        if torch.equal(prop, labels):
            break
        labels = prop
    _, cl = torch.unique(labels, return_inverse=True)
    return cl


class SuperpointTransformerSeg(nn.Module):
    def __init__(self, nc, in_channels, flags, sp_partition_knn=10, sp_tau_normal=0.95,
                sp_tau_geom=0.10, sp_partition_iters=12, sp_knn=12, sp_dim=96, sp_blocks=4):
        super().__init__()
        d = sp_dim
        self.desc = nn.Sequential(nn.Linear(in_channels * 2 + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([SPBlock(d, sp_knn) for _ in range(sp_blocks)])
        self.refine = nn.Sequential(nn.Linear(in_channels + d, 128), nn.ReLU(),
                                    nn.Dropout(0.4), nn.Linear(128, nc))
        self._normal_idx = slice(4, 7)
        geom_idx = []
        i = 7
        for flag in ("use_linearity", "use_planarity", "use_sphericity",
                    "use_verticality", "use_eigenentropy"):
            if flags.get(flag, False):
                geom_idx.append(i)
                i += 1
        self._geom_idx = geom_idx
        self.k_partition = sp_partition_knn
        self.tau_normal  = sp_tau_normal
        self.tau_geom    = sp_tau_geom
        self.partition_iters = sp_partition_iters

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        normals = flat[:, self._normal_idx]
        geom = flat[:, self._geom_idx] if self._geom_idx else None
        cl = _geometric_partition(p0, normals, geom, b0, self.k_partition,
                                  self.tau_normal, self.tau_geom, self.partition_iters)
        n_sp = int(cl.max().item()) + 1
        f_mean  = scatter_mean(flat, cl, dim=0, dim_size=n_sp)
        f_max,_ = scatter_max(flat, cl, dim=0, dim_size=n_sp)
        size    = scatter_add(torch.ones_like(cl, dtype=flat.dtype), cl,
                              dim=0, dim_size=n_sp).unsqueeze(1)
        h  = self.desc(torch.cat([f_mean, f_max, size.log1p()], 1))
        sp_pos   = scatter_mean(p0, cl, dim=0, dim_size=n_sp)
        sp_batch = scatter_max(b0, cl, dim=0, dim_size=n_sp)[0]
        for blk in self.blocks:
            h = blk(h, sp_pos, sp_batch)
        logits = self.refine(torch.cat([flat, h[cl]], 1))
        return logits.view(B, N, -1).permute(0, 2, 1)


def _fps(pos, batch, ratio):
    B = int(batch.max().item()) + 1
    out_idx = []
    for b in range(B):
        mask = (batch == b).nonzero(as_tuple=True)[0]
        n = mask.size(0)
        m = max(int(n * ratio), 1)
        pts = pos[mask]
        sel = torch.zeros(m, dtype=torch.long, device=pos.device)
        dist = torch.full((n,), float("inf"), device=pos.device)
        cur = 0
        for i in range(m):
            sel[i] = cur
            d = (pts - pts[cur]).norm(dim=1)
            dist = torch.minimum(dist, d)
            cur = int(dist.argmax().item())
        out_idx.append(mask[sel])
    return torch.cat(out_idx)


class SetAbstractionPN2(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, radius, k):
        super().__init__()
        self.ratio, self.radius, self.k = ratio, radius, k
        self.mlp = nn.Sequential(
            nn.Linear(in_ch + 3, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch))

    def forward(self, x, pos, batch):
        idx = _fps(pos, batch, self.ratio)
        c_pos, c_batch = pos[idx], batch[idx]
        e = tc_knn(pos, c_pos, self.k, batch, c_batch)
        c, nb = e[0], e[1]
        rel = pos[nb] - c_pos[c]
        feat = torch.cat([x[nb], rel], 1)
        h = self.mlp(feat)
        out, _ = scatter_max(h, c, dim=0, dim_size=c_pos.size(0))
        return out, c_pos, c_batch


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))

    def forward(self, x_coarse, pos_coarse, batch_coarse, pos_fine, batch_fine, x_skip):
        e = tc_knn(pos_coarse, pos_fine, self.k, batch_coarse, batch_fine)
        c, nb = e[0], e[1]
        d = (pos_fine[c] - pos_coarse[nb]).norm(dim=1).clamp(min=1e-8)
        w = 1.0 / d
        num = torch.zeros(pos_fine.size(0), x_coarse.size(1), device=x_coarse.device)
        den = torch.zeros(pos_fine.size(0), 1, device=x_coarse.device)
        num.index_add_(0, c, x_coarse[nb] * w.unsqueeze(1))
        den.index_add_(0, c, w.unsqueeze(1))
        interp = num / den.clamp(min=1e-8)
        if x_skip is not None:
            interp = torch.cat([interp, x_skip], 1)
        return self.mlp(interp)


class PointNetPlusPlusSeg(nn.Module):
    def __init__(self, nc, in_channels, pn2_dims=(48, 96, 192), pn2_k=16,
                pn2_ratio1=0.25, pn2_ratio2=0.25, pn2_ratio3=0.25,
                pn2_radius1=0.08, pn2_radius2=0.16, pn2_radius3=0.32):
        super().__init__()
        d = pn2_dims
        self.sa1 = SetAbstractionPN2(in_channels - 3, d[0], pn2_ratio1, pn2_radius1, pn2_k)
        self.sa2 = SetAbstractionPN2(d[0], d[1], pn2_ratio2, pn2_radius2, pn2_k)
        self.sa3 = SetAbstractionPN2(d[1], d[2], pn2_ratio3, pn2_radius3, pn2_k)
        self.fp3 = FeaturePropagation(d[2] + d[1], d[1])
        self.fp2 = FeaturePropagation(d[1] + d[0], d[0])
        self.fp1 = FeaturePropagation(d[0] + (in_channels - 3), d[0])
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        f0   = flat[:, 3:].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        f1, p1, b1 = self.sa1(f0, p0, b0)
        f2, p2, b2 = self.sa2(f1, p1, b1)
        f3, p3, b3 = self.sa3(f2, p2, b2)
        u2 = self.fp3(f3, p3, b3, p2, b2, f2)
        u1 = self.fp2(u2, p2, b2, p1, b1, f1)
        u0 = self.fp1(u1, p1, b1, p0, b0, f0)
        return self.head(u0).view(B, N, -1).permute(0, 2, 1)


class InvResMLP(nn.Module):
    def __init__(self, ch, k, expansion=4):
        super().__init__()
        self.k = k
        hidden = ch * expansion
        self.pre = nn.Sequential(nn.Linear(ch + 3, hidden), nn.ReLU())
        self.post = nn.Sequential(nn.Linear(hidden, ch))
        self.act = nn.ReLU()

    def forward(self, x, pos, batch):
        e = tc_knn(pos, pos, self.k, batch, batch)
        c, nb = e[0], e[1]
        rel = pos[nb] - pos[c]
        feat = torch.cat([x[nb], rel], 1)
        h = self.pre(feat)
        h_max, _ = scatter_max(h, c, dim=0, dim_size=x.size(0))
        h_out = self.post(h_max)
        return self.act(x + h_out)


class SetAbstractionPNX(nn.Module):
    def __init__(self, in_ch, out_ch, ratio, k):
        super().__init__()
        self.ratio, self.k = ratio, k
        self.mlp = nn.Sequential(nn.Linear(in_ch + 3, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))

    def forward(self, x, pos, batch):
        idx = _fps(pos, batch, self.ratio)
        c_pos, c_batch = pos[idx], batch[idx]
        e = tc_knn(pos, c_pos, self.k, batch, c_batch)
        c, nb = e[0], e[1]
        rel = pos[nb] - c_pos[c]
        feat = torch.cat([x[nb], rel], 1)
        h = self.mlp(feat)
        out, _ = scatter_max(h, c, dim=0, dim_size=c_pos.size(0))
        return out, c_pos, c_batch


class PointNeXtSeg(nn.Module):
    def __init__(self, nc, in_channels, pnx_dims=(48, 96, 192), pnx_k=16,
                pnx_ratio1=0.25, pnx_ratio2=0.25, pnx_ratio3=0.25):
        super().__init__()
        d = pnx_dims; k = pnx_k
        self.stem = nn.Sequential(nn.Linear(in_channels - 3, d[0]), nn.ReLU())
        self.res0 = InvResMLP(d[0], k)
        self.sa1 = SetAbstractionPNX(d[0], d[1], pnx_ratio1, k)
        self.res1 = InvResMLP(d[1], k)
        self.sa2 = SetAbstractionPNX(d[1], d[2], pnx_ratio2, k)
        self.res2 = InvResMLP(d[2], k)
        self.sa3 = SetAbstractionPNX(d[2], d[2] * 2, pnx_ratio3, k)
        self.res3 = InvResMLP(d[2] * 2, k)
        self.fp3 = FeaturePropagation(d[2] * 2 + d[2], d[2])
        self.fp2 = FeaturePropagation(d[2] + d[1], d[1])
        self.fp1 = FeaturePropagation(d[1] + d[0], d[0])
        self.head = nn.Sequential(nn.LayerNorm(d[0]), nn.Linear(d[0], 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        B, C, N = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C).contiguous()
        p0   = flat[:, :3].contiguous()
        f0   = flat[:, 3:].contiguous()
        b0   = torch.arange(B, device=x.device).repeat_interleave(N)
        f0 = self.stem(f0)
        f0 = self.res0(f0, p0, b0)
        f1, p1, b1 = self.sa1(f0, p0, b0)
        f1 = self.res1(f1, p1, b1)
        f2, p2, b2 = self.sa2(f1, p1, b1)
        f2 = self.res2(f2, p2, b2)
        f3, p3, b3 = self.sa3(f2, p2, b2)
        f3 = self.res3(f3, p3, b3)
        u2 = self.fp3(f3, p3, b3, p2, b2, f2)
        u1 = self.fp2(u2, p2, b2, p1, b1, f1)
        u0 = self.fp1(u1, p1, b1, p0, b0, f0)
        return self.head(u0).view(B, N, -1).permute(0, 2, 1)


ARCH_BUILDERS = {
    "ptv2": lambda nc, in_ch, hp, flags: PTv2Seg(
        nc, in_ch,
        dims=tuple(hp.get("model_dims", (48, 96, 192))),
        k=hp.get("model_k", 16),
        g=hp.get("model_groups", 6)),
    "kpconv": lambda nc, in_ch, hp, flags: KPConvSeg(
        nc, in_ch,
        kp_kernel_points=hp.get("kp_kernel_points", 15),
        kp_neighbors=hp.get("kp_neighbors", 16),
        kp_radius=hp.get("kp_radius", 0.06),
        kp_dims=tuple(hp.get("model_dims", hp.get("kp_dims", (48, 96, 192))))),
    "randlanet": lambda nc, in_ch, hp, flags: RandLANetSeg(
        nc, in_ch,
        rl_neighbors=hp.get("rl_neighbors", 16),
        rl_decimation=hp.get("rl_decimation", 4),
        rl_dims=tuple(hp.get("rl_dims", (48, 96, 192)))),
    "spt": lambda nc, in_ch, hp, flags: SuperpointTransformerSeg(
        nc, in_ch, flags,
        sp_partition_knn=hp.get("sp_partition_knn", 10),
        sp_tau_normal=hp.get("sp_tau_normal", 0.95),
        sp_tau_geom=hp.get("sp_tau_geom", 0.10),
        sp_partition_iters=hp.get("sp_partition_iters", 12),
        sp_knn=hp.get("sp_knn", 12),
        sp_dim=hp.get("sp_dim", 96),
        sp_blocks=hp.get("sp_blocks", 4)),
    "pointnet2": lambda nc, in_ch, hp, flags: PointNetPlusPlusSeg(
        nc, in_ch,
        pn2_dims=tuple(hp.get("pn2_dims", (48, 96, 192))),
        pn2_k=hp.get("pn2_k", 16),
        pn2_ratio1=hp.get("pn2_ratio1", 0.25),
        pn2_ratio2=hp.get("pn2_ratio2", 0.25),
        pn2_ratio3=hp.get("pn2_ratio3", 0.25),
        pn2_radius1=hp.get("pn2_radius1", 0.08),
        pn2_radius2=hp.get("pn2_radius2", 0.16),
        pn2_radius3=hp.get("pn2_radius3", 0.32)),
    "pointnext": lambda nc, in_ch, hp, flags: PointNeXtSeg(
        nc, in_ch,
        pnx_dims=tuple(hp.get("pnx_dims", (48, 96, 192))),
        pnx_k=hp.get("pnx_k", 16),
        pnx_ratio1=hp.get("pnx_ratio1", 0.25),
        pnx_ratio2=hp.get("pnx_ratio2", 0.25),
        pnx_ratio3=hp.get("pnx_ratio3", 0.25)),
}


def load_model(path, arch):
    if arch not in ARCH_BUILDERS:
        raise ValueError(f"Unknown architecture {arch!r}. Choices: {list(ARCH_BUILDERS)}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{path} is not a training checkpoint dict")

    in_ch = ckpt.get("in_channels", 7)
    nc    = int(ckpt.get("num_classes", 2))
    n_pts = int(ckpt.get("num_points", 16384))
    hp    = ckpt.get("model_hparams", {})
    flags = {k: bool(ckpt.get(k, False)) for k in
            ("use_linearity", "use_planarity", "use_sphericity",
             "use_verticality", "use_eigenentropy", "use_height_std",
             "use_height_range", "use_curvature", "use_roughness",
             "use_density", "use_mean_distance", "use_rgb")}

    model = ARCH_BUILDERS[arch](nc, in_ch, hp, flags)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    miou = ckpt.get("best_val_miou", None)
    print(f"[{arch}] loaded {path} | in_channels={in_ch}"
         f" | flags={flags} | {'val mIoU=' + f'{miou:.4f}' if miou is not None else ''}")
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


def _geom_features(points, radius, min_neighbors=8, want_linear=True, want_planar=False,
                   want_sphere=False, want_vert=False, want_entropy=False,
                   want_height_std=False, want_height_range=False,
                   want_curvature=False, want_roughness=False):
    n = len(points)
    lin    = np.zeros(n, np.float32) if want_linear else None
    pla    = np.zeros(n, np.float32) if want_planar else None
    sph    = np.zeros(n, np.float32) if want_sphere else None
    vert   = np.zeros(n, np.float32) if want_vert else None
    ent    = np.zeros(n, np.float32) if want_entropy else None
    hstd   = np.zeros(n, np.float32) if want_height_std else None
    hrange = np.zeros(n, np.float32) if want_height_range else None
    curv   = np.zeros(n, np.float32) if want_curvature else None
    rough  = np.zeros(n, np.float32) if want_roughness else None
    if not (want_linear or want_planar or want_sphere or want_vert or want_entropy
            or want_height_std or want_height_range or want_curvature or want_roughness):
        return lin, pla, sph, vert, ent, hstd, hrange, curv, rough
    need_evecs = want_vert or want_roughness
    tree = cKDTree(points)
    lists = tree.query_ball_point(points, r=radius, workers=-1)
    EIG_FLOOR = 1e-6
    for i, nbrs in enumerate(lists):
        if len(nbrs) < min_neighbors: continue
        nb = points[nbrs]
        if want_height_std or want_height_range:
            nb_z = nb[:, 2]
            if want_height_std:   hstd[i] = nb_z.std()
            if want_height_range: hrange[i] = nb_z.max() - nb_z.min()
        centroid = nb.mean(0)
        c = nb - centroid
        cov = (c.T @ c) / len(nbrs)
        if need_evecs:
            evals, evecs = np.linalg.eigh(cov)
            l3, l2, l1 = evals
        else:
            l3, l2, l1 = np.linalg.eigvalsh(cov)
        if l1 < EIG_FLOOR: continue
        l1c = l1
        if want_linear:    lin[i] = (l1 - l2) / l1c
        if want_planar:    pla[i] = (l2 - l3) / l1c
        if want_sphere:    sph[i] = l3 / l1c
        if want_curvature:
            s_sum = l1 + l2 + l3
            curv[i] = l3 / max(s_sum, 1e-9)
        if want_vert or want_roughness:
            normal_proxy = evecs[:, 0]
            if want_vert:      vert[i] = 1.0 - abs(normal_proxy[2])
            if want_roughness: rough[i] = abs((points[i] - centroid) @ normal_proxy)
        if want_entropy:
            s = l1 + l2 + l3
            if s > 1e-9:
                p = np.clip(np.array([l1, l2, l3]) / s, 1e-12, None)
                ent[i] = -(p * np.log(p)).sum()
    return lin, pla, sph, vert, ent, hstd, hrange, curv, rough
def _density_features(points, k=16, want_density=False, want_mean_distance=False):
    n = len(points)
    density = np.zeros(n, np.float32) if want_density else None
    mean_distance = np.zeros(n, np.float32) if want_mean_distance else None
    if not (want_density or want_mean_distance) or n < 2:
        return density, mean_distance
    tree = cKDTree(points)
    k_eff = min(k, n - 1)
    dist_k, _ = tree.query(points, k=k_eff + 1)
    dist_k = dist_k[:, 1:]
    if want_mean_distance:
        mean_distance[:] = dist_k.mean(axis=1)
    if want_density:
        local_radius = dist_k.max(axis=1).clip(min=1e-9)
        raw_density = k_eff / ((4 / 3) * np.pi * local_radius ** 3)
        density[:] = np.log1p(raw_density)
    return density, mean_distance


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

    want_any = any(flags.get(k, False) for k in
                   ("use_linearity", "use_planarity", "use_sphericity",
                    "use_verticality", "use_eigenentropy", "use_height_std",
                    "use_height_range", "use_curvature", "use_roughness"))
    if want_any:
        lin, pla, sph, vert, ent, hstd, hrange, curv, rough = _geom_features(
            points, radius=PREP["geom_radius"],
            want_linear=flags.get("use_linearity", False),
            want_planar=flags.get("use_planarity", False),
            want_sphere=flags.get("use_sphericity", False),
            want_vert=flags.get("use_verticality", False),
            want_entropy=flags.get("use_eigenentropy", False),
            want_height_std=flags.get("use_height_std", False),
            want_height_range=flags.get("use_height_range", False),
            want_curvature=flags.get("use_curvature", False),
            want_roughness=flags.get("use_roughness", False))
        if flags.get("use_linearity"):    cols.append(lin[:, None])
        if flags.get("use_planarity"):    cols.append(pla[:, None])
        if flags.get("use_sphericity"):   cols.append(sph[:, None])
        if flags.get("use_verticality"):  cols.append(vert[:, None])
        if flags.get("use_eigenentropy"): cols.append(ent[:, None])
        if flags.get("use_height_std"):   cols.append(hstd[:, None])
        if flags.get("use_height_range"): cols.append(hrange[:, None])
        if flags.get("use_curvature"):    cols.append(curv[:, None])
        if flags.get("use_roughness"):    cols.append(rough[:, None])

    want_density = flags.get("use_density", False)
    want_mdist   = flags.get("use_mean_distance", False)
    if want_density or want_mdist:
        dens, mdist = _density_features(
            points, k=PREP.get("density_k", 16),
            want_density=want_density, want_mean_distance=want_mdist)
        if want_density: cols.append(dens[:, None])
        if want_mdist:   cols.append(mdist[:, None])

    if flags.get("use_rgb", False):
        if rgb is None:
            rgb = np.zeros((len(points), 3), np.float32)
        cols.append(rgb.astype(np.float32))

    feat = np.column_stack(cols)
    if feat.shape[1] != in_channels:
        raise ValueError(f"Assembled {feat.shape[1]} channels but expected {in_channels}")
    feat = np.clip(feat, -100.0, 100.0)
    if not np.isfinite(feat).all():
        feat = np.nan_to_num(feat, nan=0.0, posinf=100.0, neginf=-100.0)
    return feat


@torch.no_grad()
def predict_full_cloud(model, path, num_points, in_channels, flags, overlap_factor=2):
    pts, rgb, lbl = load_pointcloud(path, return_rgb=True)
    lbl = np.zeros(len(pts), np.int64) if lbl is None else lbl.astype(np.int64)
    print(f"{os.path.basename(path)}: {len(pts):,} raw points")

    v = PREP["voxel_size"]
    if v and v > 0 and len(pts) > 0:
        keep = _voxel_keep(pts, v)
        pts, lbl = pts[keep], lbl[keep]
        if rgb is not None: rgb = rgb[keep]

    if PREP["sor_enabled"] and len(pts) > 0:
        keep = _sor_keep(pts, k=PREP["sor_k"], std_ratio=PREP["sor_std"])
        n_removed = int((~keep).sum())
        if n_removed: print(f"  SOR: removed {n_removed} outlier pts")
        pts, lbl = pts[keep], lbl[keep]
        if rgb is not None: rgb = rgb[keep]

    if PREP["cluster_filter_enabled"] and len(pts) > 0:
        keep = _cluster_keep(pts, eps=PREP["cluster_eps"], min_samples=8,
                             min_cluster_size=PREP["cluster_min_size"])
        n_removed = int((~keep).sum())
        if n_removed: print(f"  Cluster-filter: removed {n_removed} pts")
        pts, lbl = pts[keep], lbl[keep]
        if rgb is not None: rgb = rgb[keep]

    if len(pts) == 0:
        raise ValueError(
            "All points were removed by preprocessing (voxel/SOR/cluster-filter). "
            "This file may be too sparse or too small for the current PREP settings — "
            "try relaxing PREP['sor_std'], PREP['cluster_min_size'], or disabling "
            "PREP['cluster_filter_enabled'] for this file.")

    print(f"  {len(pts):,} points after preprocessing")
    feat = make_features(pts, rgb, in_channels, flags)

    n = len(feat)
    N = min(num_points, n)
    model.to(DEVICE)
    B = BATCH_SIZE

    if n <= N:
        pad_idx = np.random.RandomState(0).choice(n, N, replace=True)
        x = torch.from_numpy(feat[pad_idx][None].transpose(0, 2, 1)).to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy().ravel()
        preds = np.zeros(n, np.int64)
        preds[pad_idx] = out
        model.to("cpu")
        return pts, preds, lbl

    tree = cKDTree(pts)
    covered = np.zeros(n, bool)
    vote_sum = np.zeros((n, 2), np.int32)
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
            print(f"\r  inference: {covered.sum():,}/{n:,} pts", end="")
    flush()
    print(f"\r  inference done: {covered.sum():,}/{n:,} pts covered          ")

    if not covered.all() and covered.any():
        missing = np.flatnonzero(~covered)
        covered_idx = np.flatnonzero(covered)
        _, nn_local = cKDTree(pts[covered_idx]).query(pts[missing], k=1)
        vote_sum[missing] = vote_sum[covered_idx[nn_local]]

    preds = vote_sum.argmax(axis=1).astype(np.int64)
    model.to("cpu")
    return pts, preds, lbl


def visualize_segmentation(points, predictions, title="Segmentation",
                           labels=None, target_class=TARGET_CLASS):
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[predictions == target_class] = [0.0, 0.8, 0.0]
    colors[predictions != target_class] = [0.8, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    center = points.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector((points - center).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    span = float((points.max(0) - points.min(0)).max()) * 0.5
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=span)

    n_target = int((predictions == target_class).sum())
    n_other  = len(predictions) - n_target
    pct      = n_target / max(len(predictions), 1) * 100
    full_title = (f"{title}  |  green(target)={n_target:,} ({pct:.1f}%)  "
                  f"red(others)={n_other:,}")

    has_labels = labels is not None and (labels.any() or (labels == 0).sum() < len(labels))
    if has_labels:
        tp = int(((labels == target_class) & (predictions == target_class)).sum())
        fp = int(((labels != target_class) & (predictions == target_class)).sum())
        fn = int(((labels == target_class) & (predictions != target_class)).sum())
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        full_title += f"  |  IoU={iou:.4f}  Dice={dice:.4f}"

    print(full_title)
    o3d.visualization.draw_geometries([pcd, axes], window_name=full_title,
                                      width=1280, height=800)


def report_iou(names, true_list, pred_list, target_class=TARGET_CLASS):
    header = (f"{'File':<22}{'IoU':>7}{'Dice':>7}{'Prec':>7}{'Rec':>7}"
             f"{'TP':>9}{'FP':>9}{'FN':>9}{'TN':>9}{'GT_Wood':>10}{'Pred_Wood':>11}")
    print(f"\n{'='*len(header)}")
    print(header)
    print('-'*len(header))
    ious, dices, precs, recs = [], [], [], []
    for name, true, pred in zip(names, true_list, pred_list):
        pred_wood = int((pred == target_class).sum())
        if true is None:
            print(f"{name:<22}{'-':>7}{'-':>7}{'-':>7}{'-':>7}"
                 f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'-':>10}{pred_wood:>11,}")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=list(ARCH_BUILDERS))
    ap.add_argument("checkpoint")
    ap.add_argument("las_path")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip opening the Open3D window (report only)")
    args = ap.parse_args()

    model, n_pts, in_ch, nc, flags = load_model(args.checkpoint, args.arch)
    pts, preds, lbl = predict_full_cloud(model, args.las_path, n_pts, in_ch, flags)

    name = os.path.splitext(os.path.basename(args.las_path))[0]
    has_labels = lbl.any() or (lbl == 0).sum() < len(lbl)
    report_iou([name], [lbl if has_labels else None], [preds])

    if not args.no_viz:
        visualize_segmentation(pts, preds, title=f"INFERENCE | {args.arch} | {name}",
                               labels=lbl if has_labels else None)
