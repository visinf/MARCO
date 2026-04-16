"""Dense flow estimation, clustering, and match mining on GPU."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
from torch_kmeans.clustering.kmeans import KMeans


# ── Dense flow ────────────────────────────────────────────────────────────────

def compute_dense_flow(
    fmaps: torch.Tensor,
    masks: torch.Tensor,
    keypoints: torch.Tensor = None,
    ratio_thresh: float = 0.9,
    matmul_chunk: int = 4096,
) -> torch.Tensor:
    """Compute dense displacement field (img1 → img2) via mutual
    matching + Delaunay piecewise-affine interpolation.

    Args:
        fmaps:  [2, C, H, W] feature maps for the image pair.
        masks:  [2, H, W] bool masks on the feature-map grid.
        keypoints: [2, K, 2] (x, y) in feature-map coords (optional hard constraints).
        ratio_thresh: Lowe ratio test threshold.
        matmul_chunk: chunk size for streaming dot-products.

    Returns:
        flow: [H, W, 2] float32 displacement in feature-map pixels; zeros outside mask.
    """
    assert fmaps.ndim == 4 and fmaps.shape[0] == 2
    _, C, H, W = fmaps.shape
    device = fmaps.device

    def _l2norm(feat):
        f = feat.view(C, -1)
        f = F.normalize(f, dim=0, eps=1e-8)
        return f.view(C, H, W)

    f1 = _l2norm(fmaps[0].float())
    f2 = _l2norm(fmaps[1].float())

    assert masks.shape == (2, H, W)
    m1 = masks[0].to(device=device, dtype=torch.bool)
    m2 = masks[1].to(device=device, dtype=torch.bool)
    if not m1.any() or not m2.any():
        raise ValueError()

    y1_idx, x1_idx = torch.nonzero(m1, as_tuple=True)
    y2_idx, x2_idx = torch.nonzero(m2, as_tuple=True)

    F1 = f1[:, y1_idx, x1_idx].T.contiguous()
    F2 = f2[:, y2_idx, x2_idx].T.contiguous()
    N1, N2 = F1.shape[0], F2.shape[0]

    # Streaming NN search (avoids full N1×N2 matrix)
    best1_val = torch.full((N1,), -1e9, device=device)
    best1_idx = torch.full((N1,), -1, dtype=torch.long, device=device)
    best2_val = torch.full((N1,), -1e9, device=device)
    best2_idx = torch.full((N1,), -1, dtype=torch.long, device=device)
    for j0 in range(0, N2, matmul_chunk):
        j1 = min(N2, j0 + matmul_chunk)
        S_chunk = F1 @ F2[j0:j1].T
        nc = S_chunk.shape[1]
        if nc == 1:
            v1 = S_chunk[:, 0]
            i1 = torch.full_like(best1_idx, j0, dtype=torch.long)
            v2 = torch.full_like(v1, -1e9)
            i2 = torch.full_like(best2_idx, -1)
        else:
            v_topk, i_topk = torch.topk(S_chunk, k=2, dim=1)
            v1, v2 = v_topk[:, 0], v_topk[:, 1]
            i1, i2 = i_topk[:, 0] + j0, i_topk[:, 1] + j0

        cand_vals = torch.stack([best1_val, best2_val, v1, v2], dim=1)
        cand_idx  = torch.stack([best1_idx, best2_idx, i1, i2], dim=1)
        new_vals, which = torch.topk(cand_vals, k=2, dim=1)
        gather_idx = cand_idx.gather(1, which)
        best1_val, best2_val = new_vals[:, 0], new_vals[:, 1]
        best1_idx, best2_idx = gather_idx[:, 0], gather_idx[:, 1]

    # Row-wise ratio test
    ratio_ok = best1_val >= ratio_thresh * (best2_val + 1e-8) if N2 >= 2 \
        else torch.ones_like(best1_idx, dtype=torch.bool)

    # Column-wise top-1 (mutual NN check)
    col_best_val = torch.full((N2,), -1e9, device=device)
    col_best_i   = torch.full((N2,), -1, dtype=torch.long, device=device)
    for i0 in range(0, N1, matmul_chunk):
        i1_end = min(N1, i0 + matmul_chunk)
        S_chunk = F1[i0:i1_end] @ F2.T
        v_chunk, i_chunk = torch.max(S_chunk, dim=0)
        upd = v_chunk > col_best_val
        col_best_val[upd] = v_chunk[upd]
        col_best_i[upd]   = i_chunk[upd] + i0

    nn_mask = (col_best_i[best1_idx] == torch.arange(N1, device=device))
    keep = ratio_ok & nn_mask
    if keep.sum() < 3:
        keep = nn_mask
    kept = torch.nonzero(keep, as_tuple=True)[0]
    j_dst = best1_idx[keep]
    if kept.numel() < 3:
        raise ValueError()

    src_xy = torch.stack([x1_idx[kept], y1_idx[kept]], dim=1).float().cpu().numpy()
    dst_xy = torch.stack([x2_idx[j_dst], y2_idx[j_dst]], dim=1).float().cpu().numpy()

    # Add keypoints as hard constraints
    if keypoints is not None and keypoints.numel() >= 6:
        ks = keypoints[0].detach().cpu().numpy().astype(np.float32)
        kt = keypoints[1].detach().cpu().numpy().astype(np.float32)
        m1_np = m1.detach().cpu().numpy()
        m2_np = m2.detach().cpu().numpy()
        xi1 = np.clip(np.rint(ks[:, 0]).astype(int), 0, W - 1)
        yi1 = np.clip(np.rint(ks[:, 1]).astype(int), 0, H - 1)
        xi2 = np.clip(np.rint(kt[:, 0]).astype(int), 0, W - 1)
        yi2 = np.clip(np.rint(kt[:, 1]).astype(int), 0, H - 1)
        keep_k = m1_np[yi1, xi1] & m2_np[yi2, xi2]
        if np.any(keep_k):
            src_xy = np.vstack([src_xy, ks[keep_k]])
            dst_xy = np.vstack([dst_xy, kt[keep_k]])

    # Piecewise-affine warp via Delaunay triangulation
    tri = Delaunay(src_xy)

    def _simplex_affine(si):
        P = src_xy[tri.simplices[si]]
        Q = dst_xy[tri.simplices[si]]
        A = np.vstack([P.T, np.ones(3)])
        X = np.linalg.lstsq(A.T, Q[:, 0], rcond=None)[0]
        Y = np.linalg.lstsq(A.T, Q[:, 1], rcond=None)[0]
        return np.array([[X[0], X[1], X[2]], [Y[0], Y[1], Y[2]]], dtype=np.float32)

    Ms = np.stack([_simplex_affine(si) for si in range(len(tri.simplices))], axis=0)

    flow = np.zeros((H, W, 2), dtype=np.float32)
    m1_np = m1.detach().cpu().numpy()
    yy, xx = np.nonzero(m1_np)
    if yy.size > 0:
        pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        simp = tri.find_simplex(pts)
        valid = simp >= 0
        if valid.any():
            M = Ms[simp[valid]]
            pts_aug = np.concatenate([pts[valid], np.ones((valid.sum(), 1), np.float32)], axis=1)
            warped = (M @ pts_aug[..., None]).squeeze(-1)
            flow[yy[valid], xx[valid], 0] = warped[:, 0] - xx[valid]
            flow[yy[valid], xx[valid], 1] = warped[:, 1] - yy[valid]

    return torch.from_numpy(flow).to(torch.float32).to(device)


# ── Flow manipulation ─────────────────────────────────────────────────────────

def upsample_flow(flow_fmap: torch.Tensor, out_shape: tuple[int, int]) -> torch.Tensor:
    """Upsample flow from feature-map resolution to image resolution.

    Args:
        flow_fmap: [Hf, Wf, 2] displacement in feature-map pixels.
        out_shape: (Hi, Wi) target image size.
    Returns:
        [Hi, Wi, 2] rescaled flow.
    """
    Hf, Wf = flow_fmap.shape[:2]
    Hi, Wi = out_shape
    sx, sy = Wi / float(Wf), Hi / float(Hf)
    x = flow_fmap.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(Hi, Wi), mode="bilinear", align_corners=False)
    x[:, 0].mul_(sx)
    x[:, 1].mul_(sy)
    return x.squeeze(0).permute(1, 2, 0).contiguous()


def flow_to_hsv(flow: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Convert 2D flow to HSV encoding.

    Args:
        flow:  [H, W, 2] float32.
        valid: optional [H, W] bool mask.
    Returns:
        [H, W, 3] uint8 RGB tensor.
    """
    fx, fy = flow[..., 0], flow[..., 1]
    mag = torch.sqrt(fx * fx + fy * fy)
    ang = (torch.atan2(fy, fx) + torch.pi) / (2 * torch.pi)

    nonzero = mag[mag > 0]
    clip = torch.quantile(nonzero, 0.95) if nonzero.numel() > 0 else torch.tensor(1.0, device=flow.device)
    mag_n = torch.clamp(mag / (clip + 1e-6), 0, 1)

    H_c = ang
    S = torch.ones_like(H_c)
    V = mag_n

    h6 = H_c * 6.0
    i = torch.floor(h6).long()
    f = h6 - i.float()
    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))

    i_mod = i % 6
    r = torch.zeros_like(H_c)
    g = torch.zeros_like(H_c)
    b = torch.zeros_like(H_c)

    mask = i_mod == 0; r[mask], g[mask], b[mask] = V[mask], t[mask], p[mask]
    mask = i_mod == 1; r[mask], g[mask], b[mask] = q[mask], V[mask], p[mask]
    mask = i_mod == 2; r[mask], g[mask], b[mask] = p[mask], V[mask], t[mask]
    mask = i_mod == 3; r[mask], g[mask], b[mask] = p[mask], q[mask], V[mask]
    mask = i_mod == 4; r[mask], g[mask], b[mask] = t[mask], p[mask], V[mask]
    mask = i_mod == 5; r[mask], g[mask], b[mask] = V[mask], p[mask], q[mask]

    rgb = torch.stack([r, g, b], dim=-1)
    rgb = (rgb * 255).clamp(0, 255).byte()
    if valid is not None:
        rgb[~valid] = 0
    return rgb


# ── Clustering ────────────────────────────────────────────────────────────────

def _ll_spherical(n, sse, D):
    sigma2 = torch.clamp(sse / (n * D + 1e-12), min=1e-12)
    return -0.5 * n * (D * torch.log(2 * torch.pi * sigma2) + 1.0)


def _cluster_stats(X, labels, centers):
    K, D = centers.shape
    n = torch.bincount(labels, minlength=K)
    diffs = X - centers[labels]
    sse = torch.bincount(labels, weights=(diffs * diffs).sum(dim=1), minlength=K)
    return n.float(), sse


def _merge_stats(n1, c1, sse1, n2, c2, sse2):
    n = n1 + n2
    if n <= 0:
        return torch.tensor(0., device=c1.device), c1, torch.tensor(0., device=c1.device)
    c = (n1 * c1 + n2 * c2) / n
    shift = n1 * (c1 - c).pow(2).sum() + n2 * (c2 - c).pow(2).sum()
    sse = sse1 + sse2 + shift
    return n, c, sse


def kmeans_bic(X: torch.Tensor, k_init=32, k_min=2, seed=0, reassign_each=True) -> torch.Tensor:
    """K-means with BIC-based greedy cluster merging.

    Args:
        X: [N, 3] float32 tensor (e.g. normalised HSV/RGB flow features).
        k_init: initial number of clusters.
        k_min:  minimum number of clusters.
    Returns:
        labels: [N] long tensor of cluster assignments.
    """
    assert X.ndim == 2 and X.shape[1] == 3
    device = X.device
    N, D = X.shape

    k0 = int(min(max(k_init, k_min), max(2, N)))
    km = KMeans(n_clusters=k0, num_init=1, max_iter=100, tol=1e-4, verbose=False)
    res = km(X[None])
    labels = res.labels[0]
    centers = res.centers[0]

    n, sse = _cluster_stats(X, labels, centers)
    keep = n > 0
    centers, n, sse = centers[keep], n[keep], sse[keep]
    remap = torch.full((len(keep),), -1, dtype=torch.long, device=device)
    remap[keep.nonzero(as_tuple=True)[0]] = torch.arange(keep.sum(), device=device)
    labels = remap[labels]
    K = centers.size(0)

    if K <= k_min:
        return labels

    # Greedy merge loop
    while K > k_min:
        C = centers
        d2 = torch.cdist(C, C, p=2).pow(2)
        d2.fill_diagonal_(float('inf'))
        nbr = torch.argmin(d2, dim=1)

        best_delta = torch.tensor(0.0, device=device)
        best_i, best_j = -1, -1

        for i in range(K):
            j = int(nbr[i])
            if j <= i or n[i] <= 0 or n[j] <= 0:
                continue
            n_ij, c_ij, sse_ij = _merge_stats(n[i], C[i], sse[i], n[j], C[j], sse[j])
            ll_i = _ll_spherical(n[i], sse[i], D)
            ll_j = _ll_spherical(n[j], sse[j], D)
            ll_ij = _ll_spherical(n_ij, sse_ij, D)
            delta = (ll_ij - ll_i - ll_j) + 0.5 * (D + 2) * torch.log(torch.tensor(float(N), device=device))
            if delta > best_delta:
                best_delta, best_i, best_j = delta, i, j

        if best_delta <= 0:
            break

        i, j = best_i, best_j
        n_ij, c_ij, sse_ij = _merge_stats(n[i], centers[i], sse[i], n[j], centers[j], sse[j])
        mask = torch.ones(K, dtype=torch.bool, device=device)
        mask[j] = False
        centers[i] = c_ij
        centers, n, sse = centers[mask], n[mask], sse[mask]
        n[i], sse[i] = n_ij, sse_ij
        K = centers.size(0)

        if reassign_each:
            d2 = torch.cdist(X, centers, p=2).pow(2)
            labels = torch.argmin(d2, dim=1)
            n, sse = _cluster_stats(X, labels, centers)
            keep = n > 0
            centers, n, sse = centers[keep], n[keep], sse[keep]
            remap = torch.full((len(keep),), -1, dtype=torch.long, device=device)
            remap[keep.nonzero(as_tuple=True)[0]] = torch.arange(keep.sum(), device=device)
            labels = remap[labels]
            K = centers.size(0)

    if K > 1:
        km_final = KMeans(n_clusters=K, num_init=1, max_iter=50, tol=1e-4, verbose=False)
        labels = km_final.fit_predict(X[None], centers=centers[None])[0]

    return labels


# ── Match mining ──────────────────────────────────────────────────────────────

def filter_clusters_by_keypoints(
    flow_img: torch.Tensor,
    flow_valid: torch.Tensor,
    cluster_labels: torch.Tensor,
    kp_gt: torch.Tensor,
    threshold: float = 0.5,
    radius_px: int = 10,
) -> list[int]:
    """Keep only flow clusters whose warped keypoints are consistent with GT.

    Args:
        flow_img: [H, W, 2] dense flow.
        flow_valid: [H, W] bool mask of valid flow pixels.
        cluster_labels: [N_valid] cluster ID per valid pixel.
        kp_gt: [2, P, 2] ground-truth keypoints (src, trg) in (x, y).
        threshold: minimum fraction of consistent keypoints.
        radius_px: dilation radius for membership check.
    Returns:
        List of valid cluster IDs.
    """
    device = flow_img.device
    H, W = flow_img.shape[:2]
    ys, xs = torch.nonzero(flow_valid, as_tuple=True)
    assert cluster_labels.numel() == ys.numel()

    def _to_pix(xy):
        return xy[:, 0].round().clamp(0, W - 1).long(), xy[:, 1].round().clamp(0, H - 1).long()

    xi1, yi1 = _to_pix(kp_gt[0])
    xi2, yi2 = _to_pix(kp_gt[1])

    u, v = flow_img[..., 0][ys, xs], flow_img[..., 1][ys, xs]
    xd = (xs.float() + u).round().clamp(0, W - 1).long()
    yd = (ys.float() + v).round().clamp(0, H - 1).long()

    kernel = torch.ones((1, 1, 2 * radius_px + 1, 2 * radius_px + 1), device=device) if radius_px > 0 else None

    valid_ids = []
    for cid in torch.unique(cluster_labels):
        sel = (cluster_labels == cid).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue

        src_mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        dst_mask = torch.zeros_like(src_mask)
        src_mask[0, 0, ys[sel], xs[sel]] = 1.0
        dst_mask[0, 0, yd[sel], xd[sel]] = 1.0

        if kernel is not None:
            src_mask = (F.conv2d(src_mask, kernel, padding=radius_px) > 0).float()
            dst_mask = (F.conv2d(dst_mask, kernel, padding=radius_px) > 0).float()

        in_src = src_mask[0, 0].bool()[yi1, xi1]
        n_in_src = in_src.sum().item()
        if n_in_src == 0:
            continue
        in_dst = dst_mask[0, 0].bool()[yi2, xi2]
        frac = (in_src & in_dst).float().sum().item() / n_in_src

        if frac >= threshold:
            valid_ids.append(int(cid.item()))

    return valid_ids


def collect_matches(
    flow_img: torch.Tensor,
    flow_valid: torch.Tensor,
    cluster_labels: torch.Tensor,
    valid_cluster_ids,
    object_masks: torch.Tensor,
    max_points_per_cluster: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Collect source→target match pairs from flow, filtered by cluster and object mask.

    Args:
        flow_img: [H, W, 2] dense flow.
        flow_valid: [H, W] bool.
        cluster_labels: [N_valid] cluster ID per valid pixel.
        valid_cluster_ids: iterable of cluster IDs to keep.
        object_masks: [2, H, W] bool masks for source/target.
        max_points_per_cluster: optional cap per cluster.
    Returns:
        [2, N, 2] int32 tensor — [0]=source (x,y), [1]=target (x,y).
    """
    device = flow_img.device
    H, W = flow_img.shape[:2]
    obj2 = object_masks[1].bool()

    ys, xs = torch.nonzero(flow_valid, as_tuple=True)
    u = flow_img[..., 0][ys, xs]
    v = flow_img[..., 1][ys, xs]
    xd = (xs.float() + u).round().clamp(0, W - 1).long()
    yd = (ys.float() + v).round().clamp(0, H - 1).long()

    gen = torch.Generator(device=device).manual_seed(seed)
    src_list, dst_list = [], []

    for cid in valid_cluster_ids:
        sel = (cluster_labels == int(cid)).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue

        x1, y1 = xs[sel], ys[sel]
        x2, y2 = xd[sel], yd[sel]

        keep = obj2[y2, x2]
        if not keep.any():
            continue
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]

        if max_points_per_cluster is not None and x1.numel() > max_points_per_cluster:
            idx = torch.randperm(x1.numel(), generator=gen, device=device)[:max_points_per_cluster]
            x1, y1, x2, y2 = x1[idx], y1[idx], x2[idx], y2[idx]

        src_list.append(torch.stack([x1, y1], dim=1))
        dst_list.append(torch.stack([x2, y2], dim=1))

    if not src_list:
        return torch.zeros((2, 0, 2), dtype=torch.int32, device=device)

    src_all = torch.cat(src_list, dim=0)
    dst_all = torch.cat(dst_list, dim=0)
    return torch.stack([src_all, dst_all], dim=0).to(torch.int32)
