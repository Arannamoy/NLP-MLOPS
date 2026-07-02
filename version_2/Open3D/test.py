import numpy as np
import open3d as o3d
import os
import csv
import json
import warnings
from typing import Tuple, Dict, Any, List
import laspy

# Suppress Open3D warnings for clean console output during batch processing
warnings.filterwarnings("ignore")


def generate_heap_geometry(xx: np.ndarray, yy: np.ndarray, shape_type: str, radius: float, max_height: float) -> np.ndarray:
    """Generates the mathematical base geometry of the wood powder pile."""
    r_sq = xx**2 + yy**2
    r_dist = np.sqrt(r_sq)
    
    if shape_type == "Gaussian":
        zz = max_height * np.exp(-r_sq / (2 * (radius / 2.5)**2))
    elif shape_type == "Cone":
        zz = max_height * (1 - r_dist / radius)
        zz[zz < 0] = 0
    elif shape_type == "Truncated Cone":
        zz = max_height * (1 - r_dist / radius)
        zz = np.clip(zz, 0, max_height * 0.7)
    elif shape_type == "Ellipsoid":
        inside = 1 - r_sq / (radius**2)
        inside[inside < 0] = 0
        zz = max_height * np.sqrt(inside)
    elif shape_type == "Double Peak":
        offset = radius / 2.5
        z1 = (max_height * 0.9) * np.exp(-((xx - offset)**2 + yy**2) / (2 * (radius / 3)**2))
        z2 = (max_height * 0.8) * np.exp(-((xx + offset)**2 + yy**2) / (2 * (radius / 3)**2))
        zz = np.maximum(z1, z2)
    elif shape_type == "Triple Peak":
        o = radius / 2.5
        z1 = max_height * np.exp(-((xx - o)**2 + (yy - o)**2) / (2 * (radius / 3)**2))
        z2 = (max_height * 0.85) * np.exp(-((xx + o)**2 + (yy - o)**2) / (2 * (radius / 3)**2))
        z3 = (max_height * 0.7) * np.exp(-(xx**2 + (yy + o)**2) / (2 * (radius / 3)**2))
        zz = np.maximum.reduce([z1, z2, z3])
    elif shape_type == "Asymmetric Heap":
        zz = max_height * np.exp(-( (xx**2)/(2*(radius/1.5)**2) + (yy**2)/(2*(radius/3)**2) ))
    elif shape_type == "Random Irregular Heap":
        base = max_height * np.exp(-r_sq / (2 * (radius / 2.5)**2))
        noise = (max_height * 0.2) * np.sin(10 * xx) * np.cos(10 * yy)
        zz = base + noise * (base > max_height * 0.1)
        zz[zz < 0] = 0
    else:
        zz = np.zeros_like(xx)

    # Base surface roughness (clumping)
    uneven_noise = np.random.uniform(0.01, 0.04) * np.sin(20 * xx) * np.cos(20 * yy)
    zz += uneven_noise * (zz > 0.05)
    
    return np.clip(zz, 0, None)


def generate_terrain(xx: np.ndarray, yy: np.ndarray, terrain_type: str) -> np.ndarray:
    """Generates the underlying ground topography."""
    if terrain_type == "Flat":
        return np.zeros_like(xx)
    elif terrain_type == "Uneven":
        return 0.04 * np.sin(4 * xx) + 0.04 * np.cos(4 * yy)
    elif terrain_type == "Sloped":
        slope_x, slope_y = np.random.uniform(-0.08, 0.08, 2)
        return slope_x * xx + slope_y * yy
    elif terrain_type == "Rough industrial floor":
        return np.random.normal(0, 0.015, xx.shape)
    return np.zeros_like(xx)


def simulate_realistic_lidar(points: np.ndarray, sensor_pos: np.ndarray, az_res_deg: float, el_res_deg: float) -> np.ndarray:
    """
    Simulates a 3D LiDAR scan using spherical raycasting.
    Naturally creates distance-based point density and occlusion/shadows.
    """
    az_res = np.radians(az_res_deg)
    el_res = np.radians(el_res_deg)

    vecs = points - sensor_pos
    distances = np.linalg.norm(vecs, axis=1)

    # Filter out points too close to sensor to avoid division by zero
    valid_mask = distances > 0.1
    vecs = vecs[valid_mask]
    points = points[valid_mask]
    distances = distances[valid_mask]

    if len(points) == 0:
        return points

    # Convert to spherical coordinates for beam binning
    azimuth = np.arctan2(vecs[:, 1], vecs[:, 0])
    # Clip to strictly avoid NaN from floating point precision errors
    elevation = np.arcsin(np.clip(vecs[:, 2] / distances, -1.0, 1.0))

    az_bins = np.round(azimuth / az_res).astype(np.int32)
    el_bins = np.round(elevation / el_res).astype(np.int32)

    # Sort by distance (closest first for occlusion)
    sort_idx = np.argsort(distances)
    az_bins_s = az_bins[sort_idx]
    el_bins_s = el_bins[sort_idx]
    points_s = points[sort_idx]

    # Shift bins to positive integers for robust hashing
    az_shifted = (az_bins_s - np.min(az_bins_s)).astype(np.int64)
    el_shifted = (el_bins_s - np.min(el_bins_s)).astype(np.int64)
    hash_keys = az_shifted * 100000 + el_shifted

    # The first unique hash is the closest point in that specific beam (occluding the rest)
    _, unique_idx = np.unique(hash_keys, return_index=True)

    return points_s[unique_idx]


def inject_artifacts(points: np.ndarray, noise_level: float, max_radius: float, max_z: float) -> np.ndarray:
    """Applies sensor noise, random dropout, occlusion holes, and flying outliers."""
    if len(points) == 0:
        return points

    # 1. Random Dropout (Sparse regions)
    dropout_mask = np.random.rand(len(points)) > np.random.uniform(0.01, 0.05)
    points = points[dropout_mask]

    # 2. Gaussian Sensor Noise
    points += np.random.normal(0, noise_level, points.shape)

    # 3. Missing Regions (Occlusion holes / dead beams)
    num_holes = np.random.randint(1, 6)
    valid_mask = np.ones(len(points), dtype=bool)
    for _ in range(num_holes):
        hx = np.random.uniform(-max_radius, max_radius)
        hy = np.random.uniform(-max_radius, max_radius)
        h_radius = np.random.uniform(0.05, 0.3)
        dist_sq = (points[:, 0] - hx)**2 + (points[:, 1] - hy)**2
        valid_mask &= (dist_sq > h_radius**2)
    points = points[valid_mask]

    # 4. Outliers (Flying points / multi-path reflections)
    num_outliers = np.random.randint(20, 200)
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    outliers = np.column_stack((
        np.random.uniform(min_x, max_x, num_outliers),
        np.random.uniform(min_y, max_y, num_outliers),
        np.random.uniform(-0.2, max_z + 0.5, num_outliers)
    ))
    
    return np.vstack((points, outliers))


def is_valid_point_cloud(points: np.ndarray, min_points: int = 100) -> bool:
    """Defensive programming check to ensure validity before export."""
    if points.size == 0 or len(points) < min_points:
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    return True


def generate_synthetic_dataset(num_samples=100, output_dir="sample_data"):
    """Main generation loop with integrated defensive programming and realistic simulations."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_data = [["filename", "volume_m3"]]
    metadata_list = []
    
    shape_types = [
        "Gaussian", "Cone", "Truncated Cone", "Ellipsoid", 
        "Double Peak", "Triple Peak", "Asymmetric Heap", "Random Irregular Heap"
    ]
    terrain_types = ["Flat", "Uneven", "Sloped", "Rough industrial floor"]

    print(f"Generating {num_samples} samples in '{output_dir}'...")

    i = 0
    while i < num_samples:
        try:
            # 1. Randomize Heap and Environment Parameters
            shape_type = np.random.choice(shape_types)
            terrain_type = np.random.choice(terrain_types)
            
            radius = np.random.uniform(0.5, 2.5)   
            max_height = np.random.uniform(0.3, 1.2)
            noise_level = np.random.uniform(0.002, 0.015)
            
            rot_z = np.random.uniform(0, 2 * np.pi)
            trans_x, trans_y = np.random.uniform(-0.5, 0.5, 2)
            
            # Create a dense mathematical grid (1cm res)
            grid_res = 0.001 
            area_size = radius * 1.5
            x = np.arange(-area_size, area_size, grid_res)
            y = np.arange(-area_size, area_size, grid_res)
            xx, yy = np.meshgrid(x, y)
            
            # 2. Build Mathematics & Calculate True Volume
            heap_z = generate_heap_geometry(xx, yy, shape_type, radius, max_height)
            terrain_z = generate_terrain(xx, yy, terrain_type)
            
            # GT Volume strictly calculated from the heap's actual mathematical footprint
            cell_area = grid_res ** 2
            gt_volume = float(np.sum(heap_z * cell_area))
            
            # Combine surfaces
            total_z = terrain_z.copy()
            heap_mask = heap_z > 0.005
            total_z[heap_mask] += heap_z[heap_mask]
            
            points = np.column_stack((xx.ravel(), yy.ravel(), total_z.ravel())).astype(np.float64)

            # 3. Apply Transformations
            rot_matrix = np.array([
                [np.cos(rot_z), -np.sin(rot_z), 0],
                [np.sin(rot_z),  np.cos(rot_z), 0],
                [0,              0,             1]
            ])
            points = points @ rot_matrix.T
            points += np.array([trans_x, trans_y, 0])

            # 4. Realistic LiDAR Simulation
            sensor_pos = np.array([
                np.random.uniform(-area_size, area_size),
                np.random.uniform(-area_size, area_size),
                max_height + np.random.uniform(1.0, 3.5)
            ])
            az_res = np.random.uniform(0.1, 0.4)
            el_res = np.random.uniform(0.1, 0.4)
            
            points = simulate_realistic_lidar(points, sensor_pos, az_res, el_res)
            
            # 5. Inject Noise, Dropouts, Outliers
            points = inject_artifacts(points, noise_level, area_size, max_height)
            
            # Final array cleanup and validation
            points = np.unique(points, axis=0)
            if not is_valid_point_cloud(points):
                # If the generation corrupted the cloud (e.g., raycasting dropped all points), regenerate silently
                continue

            # 6. Open3D PointCloud Creation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate Normals for Deep Learning frameworks
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Defensive step: Ensure normals are valid (isolated points get NaN normals)
            normals_arr = np.asarray(pcd.normals)
            valid_normals = ~np.isnan(normals_arr).any(axis=1)
            pcd = pcd.select_by_index(np.where(valid_normals)[0])
            pcd.orient_normals_towards_camera_location(camera_location=sensor_pos)
            
            # Final pre-save check
            if not is_valid_point_cloud(np.asarray(pcd.points)):
                continue

            # 7. Export files to .las format
            filename = f"sample_{i:04d}.las"
            filepath = os.path.join(output_dir, filename)
            
            # Extract raw points from the Open3D object
            points_np = np.asarray(pcd.points)
            
            # Create standard LAS header (1mm precision)
            header = laspy.LasHeader(point_format=0, version="1.2")
            header.offsets = np.min(points_np, axis=0)
            header.scales = np.array([0.001, 0.001, 0.001])
            
            # Create LAS file and write XYZ coordinates
            las = laspy.LasData(header)
            las.x = points_np[:, 0]
            las.y = points_np[:, 1]
            las.z = points_np[:, 2]
            
            las.write(filepath)

            # Collect data for logs
            csv_data.append([filename, f"{gt_volume:.6f}"])
            
            metadata_list.append({
                "filename": filename,
                "shape_type": shape_type,
                "terrain_type": terrain_type,
                "height_m": round(max_height, 4),
                "radius_m": round(radius, 4),
                "rotation_z_rad": round(rot_z, 4),
                "translation_xy": [round(trans_x, 4), round(trans_y, 4)],
                "point_count": len(pcd.points),
                "noise_level_m": round(noise_level, 4),
                "volume_m3": round(gt_volume, 6),
                "sensor_position": [round(c, 4) for c in sensor_pos],
                "scan_resolution_deg": [round(az_res, 3), round(el_res, 3)]
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_samples} samples...")
            
            i += 1 # Successfully generated, move to next index

        except Exception as e:
            # Catch unexpected math/io errors to prevent breaking the whole batch
            print(f"Warning: Failed to generate sample {i} due to: {str(e)}. Retrying...")
            continue

    # 8. Save CSV and JSON Metadata
    csv_path = os.path.join(output_dir, "volume.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, mode="w") as f:
        json.dump(metadata_list, f, indent=4)
        
    print(f"Dataset generation complete! Files saved to '{output_dir}'.")


if __name__ == "__main__":
    # Seed for reproducibility across generation runs
    np.random.seed(42)
    generate_synthetic_dataset(num_samples=2000, output_dir="sample_data")