import sys
import numpy as np
import laspy
from scipy.spatial import cKDTree

WANTED_FEATURES = [
    "z", "relative_height", "height_std", "height_range", "density",
    "mean_distance", "linearity", "planarity", "curvature", "roughness",
    "verticality",
]


def check_available_dimensions(las_path):
    las = laspy.read(las_path)
    dims = set(las.point_format.dimension_names)
    print(f"Dimensions actually stored in {las_path}:")
    for d in sorted(dims):
        print(f"  {d}")
    print()
    print("Checking against the requested feature list:")
    missing = []
    for w in WANTED_FEATURES:
        found = w in dims
        print(f"  {w:18s}: {'FOUND in file' if found else 'NOT in file -> will compute'}")
        if not found:
            missing.append(w)
    return missing


def compute_all_features(points, radius=0.1, k=16, min_neighbors=8):
    n = len(points)
    z = points[:, 2]
    relative_height = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    tree = cKDTree(points)

    dist_k, _ = tree.query(points, k=k + 1)
    dist_k = dist_k[:, 1:]
    mean_distance = dist_k.mean(axis=1)
    local_radius = dist_k.max(axis=1).clip(min=1e-9)
    density = k / ((4 / 3) * np.pi * local_radius ** 3)

    lists = tree.query_ball_point(points, r=radius, workers=-1)

    height_std = np.zeros(n, np.float32)
    height_range = np.zeros(n, np.float32)
    linearity = np.zeros(n, np.float32)
    planarity = np.zeros(n, np.float32)
    curvature = np.zeros(n, np.float32)
    roughness = np.zeros(n, np.float32)
    verticality = np.zeros(n, np.float32)

    for i, nbrs in enumerate(lists):
        if len(nbrs) < min_neighbors:
            continue
        nb_pts = points[nbrs]
        nb_z = nb_pts[:, 2]
        height_std[i] = nb_z.std()
        height_range[i] = nb_z.max() - nb_z.min()

        centroid = nb_pts.mean(0)
        c = nb_pts - centroid
        cov = (c.T @ c) / len(nbrs)
        evals, evecs = np.linalg.eigh(cov)
        l3, l2, l1 = evals
        s = l1 + l2 + l3
        l1c = max(l1, 1e-9)

        linearity[i] = (l1 - l2) / l1c
        planarity[i] = (l2 - l3) / l1c
        curvature[i] = l3 / max(s, 1e-9)

        normal = evecs[:, 0]
        verticality[i] = 1.0 - abs(normal[2])
        roughness[i] = abs((points[i] - centroid) @ normal)

    return {
        "z": z,
        "relative_height": relative_height,
        "height_std": height_std,
        "height_range": height_range,
        "density": density,
        "mean_distance": mean_distance,
        "linearity": linearity,
        "planarity": planarity,
        "curvature": curvature,
        "roughness": roughness,
        "verticality": verticality,
    }


if __name__ == "__main__":
    las_path = sys.argv[1] if len(sys.argv) > 1 else "data/test/sample_data_0046.las"

    missing = check_available_dimensions(las_path)
    print()

    if missing:
        print(f"{len(missing)} feature(s) need to be computed: {missing}")
        las = laspy.read(las_path)
        pts = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                               np.asarray(las.z)]).astype(np.float64)
        feats = compute_all_features(pts)
        print()
        print("Computed feature summary:")
        for name in WANTED_FEATURES:
            arr = feats[name]
            print(f"  {name:18s}: mean={arr.mean():.4f}  min={arr.min():.4f}  "
                  f"max={arr.max():.4f}  has_nan={bool(np.isnan(arr).any())}")
    else:
        print("All requested features are already stored in the file.")
