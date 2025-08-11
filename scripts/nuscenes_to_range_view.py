#!/usr/bin/env python3
"""
Standalone script to convert NuScenes LiDAR data to range view images.

This script extracts the core functionality from the Point Frustums codebase
to create range view representations of NuScenes LiDAR point clouds.

Usage:
    python scripts/nuscenes_to_range_view.py \
        --nuscenes_root /path/to/nuscenes \
        --output_dir /path/to/output \
        --split train \
        --version v1.0-trainval
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copious.io.fs import parent_ensured_path
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from tqdm import tqdm


def cart_to_sph_numpy(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        points: Array of shape (N, 3) with [x, y, z] coordinates
        
    Returns:
        Array of shape (N, 3) with [radius, polar, azimuthal] coordinates
    """
    # FIXME: import matplotlib.pyplot as plt; plt.hist(points[:, 2], bins=100); plt.savefig('z_hist.png')
    coos_sph = np.empty(points.shape, dtype=np.float32)
    xy = points[:, 0] ** 2 + points[:, 1] ** 2  # squared distance to observer in ground projection
    coos_sph[:, 0] = np.sqrt(xy + points[:, 2] ** 2)  # Radial distance
    coos_sph[:, 1] = np.arctan2(points[:, 2], np.sqrt(xy))  # Polar/inclination angle (from Z-axis down)
    coos_sph[:, 2] = np.arctan2(points[:, 1], points[:, 0])  # Azimuth angle
    # FIXME: import matplotlib.pyplot as plt; plt.hist(coos_sph[:, 1], bins=30); plt.savefig('elevation_hist.png')
    return coos_sph


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return (deg / 360) * 2 * np.pi


class RangeViewConverter:
    """Converts NuScenes LiDAR point clouds to range view images."""
    
    def __init__(
        self,
        n_splits_azi: int = 512,
        n_splits_pol: int = 64,
        fov_azi_deg: Tuple[int, int] = (-180, 180),
        fov_pol_deg: Tuple[int, int] = (-30, 10),
        max_distance: float = 100.0,
        min_distance: float = 1.0,
        num_sweeps: int = 1,
    ):
        """
        Initialize range view converter.
        
        Args:
            n_splits_azi: Number of azimuthal bins (horizontal resolution)
            n_splits_pol: Number of polar bins (vertical resolution)
            fov_azi_deg: Azimuthal field of view in degrees (left, right)
            fov_pol_deg: Polar field of view in degrees (down, up)
            max_distance: Maximum distance to consider
            min_distance: Minimum distance to filter noise
            num_sweeps: Number of sweeps to accumulate
        """
        self.n_splits_azi = n_splits_azi
        self.n_splits_pol = n_splits_pol
        self.fov_azi = tuple(deg_to_rad(x) for x in fov_azi_deg)
        self.fov_pol = tuple(deg_to_rad(x) for x in fov_pol_deg)
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.num_sweeps = num_sweeps
        
        # Calculate discretization parameters
        self.range_pol = self.fov_pol[1] - self.fov_pol[0]
        self.range_azi = self.fov_azi[1] - self.fov_azi[0]
        self.delta_pol = self.range_pol / self.n_splits_pol
        self.delta_azi = self.range_azi / self.n_splits_azi
        
    def load_sweeps(self, db: NuScenes, sample: dict, sensor_id: str = "LIDAR_TOP") -> np.ndarray:
        """
        Load multiple LiDAR sweeps and combine them.
        
        Args:
            db: NuScenes database instance
            sample: Sample dictionary
            sensor_id: Sensor identifier
            
        Returns:
            Combined point cloud with channels [x, y, z, intensity, timestamp, r, theta, phi]
        """
        sweeps = []
        
        # Get the reference sample_data
        ref_sample_data_token = sample["data"][sensor_id]
        sample_data = db.get("sample_data", ref_sample_data_token)
        timestamp_reference = pd.to_datetime(sample_data["timestamp"], utc=True, unit="us", origin="unix")
        
        # Get global to sensor transformation for reference frame
        ego = db.get("ego_pose", sample_data["ego_pose_token"])
        sensor = db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        
        ego2global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
        sensor2ego = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=False)
        ego2sensor = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=True)
        global2ego = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=True)
        global2sensor = np.dot(ego2sensor, global2ego)
        
        # Collect sweep files
        sample_data_token = ref_sample_data_token
        pc_files = []
        
        for _ in range(self.num_sweeps):
            file = db.get_sample_data_path(sample_data_token)
            pc_files.append((file, sample_data_token, sample_data))
            
            # Check if previous sweep exists
            if sample_data["prev"] == "":
                break
                
            sample_data_token = sample_data["prev"]
            sample_data = db.get("sample_data", sample_data_token)
        
        # Load and process sweeps
        for file, sample_data_token, sample_data in pc_files:
            try:
                pc = LidarPointCloud.from_file(file)
            except ValueError:
                print(f"Warning: Could not load {file}")
                continue
            
            # Filter points by distance
            distances = np.linalg.norm(pc.points[0:2, :], axis=0)
            pc.points = pc.points[:, distances >= self.min_distance]
            
            # Transform non-reference sweeps to reference frame
            if ref_sample_data_token != sample_data_token:
                current_ego = db.get("ego_pose", sample_data["ego_pose_token"])
                current_sensor = db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                
                current_ego2global = transform_matrix(
                    current_ego["translation"], 
                    Quaternion(current_ego["rotation"]), 
                    inverse=False
                )
                current_sensor2ego = transform_matrix(
                    current_sensor["translation"],
                    Quaternion(current_sensor["rotation"]),
                    inverse=False
                )
                sensor2global = np.dot(current_ego2global, current_sensor2ego)
                
                # Transform: sensor -> global -> reference_sensor
                pc.transform(np.dot(global2sensor, sensor2global))
            
            # Filter out points on the ego car: convert to reference ego frame, apply mask, convert back to sensor frame
            # reference sensor -> reference ego
            pc.transform(sensor2ego)
            x_ego = pc.points[0, :]
            y_ego = pc.points[1, :]
            mask = np.logical_not(
                (y_ego < 3.2) & (y_ego > -2.2) &
                (x_ego < 0.8) & (x_ego > -0.8)
            )
            pc.points = pc.points[:, mask]
            # reference ego -> reference sensor
            pc.transform(ego2sensor)
            
            # Add timestamp information
            timestamp_sample = pd.to_datetime(sample_data["timestamp"], utc=True, unit="us", origin="unix")
            time_diff = (timestamp_sample - timestamp_reference).total_seconds()
            timestamp_channel = np.full(pc.points.shape[1], time_diff, dtype=np.float32)
            
            # Add timestamp as 5th channel
            pc_with_time = np.vstack([pc.points, timestamp_channel])
            sweeps.append(pc_with_time.astype(np.float32))
        
        # Combine all sweeps
        combined = np.concatenate(sweeps, axis=1).T  # Shape: (N, 5) [x, y, z, intensity, timestamp]
        
        # Add spherical coordinates
        spherical = cart_to_sph_numpy(combined[:, :3])  # [radius, polar, azimuthal]
        
        # Final point cloud: [x, y, z, intensity, timestamp, radius, polar, azimuthal]
        return np.concatenate((combined, spherical), axis=1)
    
    def filter_fov(self, points: np.ndarray) -> np.ndarray:
        """Filter points within field of view."""
        azimuthal = points[:, 7]  # azimuthal angle
        polar = points[:, 6]      # polar angle
        
        mask = (
            (azimuthal >= self.fov_azi[0]) &
            (azimuthal < self.fov_azi[1]) &
            (polar >= self.fov_pol[0]) &
            (polar < self.fov_pol[1])
        )
        
        return points[mask]
    
    def get_frustum_indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get frustum indices for each point."""
        azimuthal = points[:, 7]  # azimuthal angle
        polar = points[:, 6]      # polar angle
        
        # Calculate bin indices
        i_azi = ((azimuthal - self.fov_azi[0]) // self.delta_azi).astype(np.int64)
        i_pol = ((polar - self.fov_pol[0]) // self.delta_pol).astype(np.int64)
        
        # Clamp to valid range
        i_azi = np.clip(i_azi, 0, self.n_splits_azi - 1)
        i_pol = np.clip(i_pol, 0, self.n_splits_pol - 1)
        
        return i_azi, i_pol
    
    def create_range_image(self, points: np.ndarray, aggregation: str = "closest") -> np.ndarray:
        """
        Create range view image from point cloud.
        
        Args:
            points: Point cloud array with shape (N, 8)
            aggregation: Aggregation method ("max", "mean", "closest")
            
        Returns:
            Range view image of shape (n_splits_pol, n_splits_azi)
        """
        # Filter points in FOV
        points = self.filter_fov(points)
        
        if len(points) == 0:
            return np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.float32)
        
        # Get frustum indices
        i_azi, i_pol = self.get_frustum_indices(points)
        distances = points[:, 5]  # radius
        
        # Create range image
        range_image = np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.float32)
        
        if aggregation == "max":
            # Take maximum distance in each bin
            for i in range(len(points)):
                current_dist = distances[i]
                existing_dist = range_image[i_pol[i], i_azi[i]]
                if current_dist > existing_dist:
                    range_image[i_pol[i], i_azi[i]] = current_dist
                    
        elif aggregation == "closest":
            # Take minimum distance in each bin
            range_image.fill(np.inf)
            for i in range(len(points)):
                current_dist = distances[i]
                existing_dist = range_image[i_pol[i], i_azi[i]]
                if current_dist < existing_dist:
                    range_image[i_pol[i], i_azi[i]] = current_dist
            range_image[range_image == np.inf] = 0
            
        elif aggregation == "mean":
            # Take mean distance in each bin
            count_image = np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.int32)
            for i in range(len(points)):
                range_image[i_pol[i], i_azi[i]] += distances[i]
                count_image[i_pol[i], i_azi[i]] += 1
            
            # Avoid division by zero
            mask = count_image > 0
            range_image[mask] /= count_image[mask]
        
        # Clip to maximum distance
        range_image = np.clip(range_image, 0, self.max_distance)
        
        return range_image
    
    def create_intensity_image(self, points: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        """
        Create intensity image from point cloud.
        
        Args:
            points: Point cloud array with shape (N, 8)
            aggregation: Aggregation method ("max", "mean")
            
        Returns:
            Intensity image of shape (n_splits_pol, n_splits_azi)
        """
        # Filter points in FOV
        points = self.filter_fov(points)
        
        if len(points) == 0:
            return np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.float32)
        
        # Get frustum indices
        i_azi, i_pol = self.get_frustum_indices(points)
        intensities = points[:, 3]  # intensity
        
        # Create intensity image
        intensity_image = np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.float32)
        
        if aggregation == "max":
            # Take maximum intensity in each bin
            for i in range(len(points)):
                current_intensity = intensities[i]
                existing_intensity = intensity_image[i_pol[i], i_azi[i]]
                if current_intensity > existing_intensity:
                    intensity_image[i_pol[i], i_azi[i]] = current_intensity
                    
        elif aggregation == "mean":
            # Take mean intensity in each bin
            count_image = np.zeros((self.n_splits_pol, self.n_splits_azi), dtype=np.int32)
            for i in range(len(points)):
                intensity_image[i_pol[i], i_azi[i]] += intensities[i]
                count_image[i_pol[i], i_azi[i]] += 1
            
            # Avoid division by zero
            mask = count_image > 0
            intensity_image[mask] /= count_image[mask]
        
        return intensity_image


def save_range_view_image(
    range_image: np.ndarray, 
    output_path: str, 
    title: str = "Range View",
    colormap: str = "viridis",
    fov_azi_deg: Tuple[float, float] = (-180.0, 180.0)
):
    """Save range view as image file with azimuth degree ticks on x-axis."""
    plt.figure(figsize=(12, 4))
    plt.imshow(range_image, aspect='auto', cmap=colormap, origin='lower')
    plt.colorbar(label='Distance (m)')
    plt.title(title)
    
    # Set azimuth ticks/labels in degrees on the x-axis
    n_cols = range_image.shape[1]
    left_deg, right_deg = float(fov_azi_deg[0]), float(fov_azi_deg[1])
    # Choose a reasonable number of ticks
    num_ticks = 9 if n_cols >= 256 else 5
    tick_labels = np.linspace(left_deg, right_deg, num=num_ticks)
    tick_positions = (tick_labels - left_deg) / (right_deg - left_deg) * (n_cols - 1)
    plt.xticks(tick_positions, [f"{int(a)}°" for a in tick_labels])
    
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (bins)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_range_view_data(
    range_image: np.ndarray,
    intensity_image: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None
):
    """Save range view data as numpy file."""
    data = {
        'range': range_image,
        'intensity': intensity_image,
    }
    if metadata is not None:
        # Store metadata as an object array to satisfy numpy typing
        data['metadata'] = np.array(metadata, dtype=object)
    
    np.savez_compressed(output_path, **data)


def main():
    parser = argparse.ArgumentParser(description="Convert NuScenes LiDAR to range view images")
    parser.add_argument("--nuscenes_root", required=True, help="Path to NuScenes dataset root")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for range view images")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split")
    parser.add_argument("--version", default="v1.0-trainval", help="NuScenes version")
    parser.add_argument("--n_splits_azi", type=int, default=512, help="Azimuthal resolution")
    parser.add_argument("--n_splits_pol", type=int, default=64, help="Polar resolution")
    parser.add_argument("--fov_azi_deg", nargs=2, type=int, default=[-180, 180], help="Azimuthal FOV in degrees")
    parser.add_argument("--fov_pol_deg", nargs=2, type=int, default=[-30, 10], help="Polar FOV in degrees")
    parser.add_argument("--max_distance", type=float, default=100.0, help="Maximum distance")
    parser.add_argument("--min_distance", type=float, default=1.0, help="Minimum distance")
    parser.add_argument("--num_sweeps", type=int, default=1, help="Number of sweeps to accumulate")
    parser.add_argument("--save_images", action="store_true", help="Save visualization images")
    parser.add_argument("--save_data", action="store_true", help="Save numpy data files")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    # Initialize NuScenes
    print(f"Loading NuScenes {args.version} from {args.nuscenes_root}")
    db = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)
    
    # Initialize converter
    converter = RangeViewConverter(
        n_splits_azi=args.n_splits_azi,
        n_splits_pol=args.n_splits_pol,
        fov_azi_deg=tuple(args.fov_azi_deg),
        fov_pol_deg=tuple(args.fov_pol_deg),
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        num_sweeps=args.num_sweeps,
    )
    
    # Get samples for the specified split
    # scene_splits = create_splits_scenes()
    
    # if args.version == "v1.0-mini":
    #     scene_names = scene_splits[f'mini_{args.split}']
    # elif args.version == "v1.0-trainval":
    #     scene_names = scene_splits[args.split]
    # else:    
    #     raise ValueError(f"Invalid split ({args.split}) and version ({args.version}) combination.")
    
    # available_scenes = sorted(set(scene_names) & set(s['name'] for s in db.scene))
    
    # Get sample tokens
    sample_tokens = []
    for scene in db.scene:
        # Iterate through samples in scene
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_tokens.append(sample_token)
            sample = db.get('sample', sample_token)
            sample_token = sample['next']
    
    if args.max_samples:
        sample_tokens = sample_tokens[:args.max_samples]
    
    print(f"Processing {len(sample_tokens)} samples from {args.split} split")
    
    # Process samples
    for i, sample_token in enumerate(tqdm(sample_tokens, desc="Processing samples")):
        sample = db.get('sample', sample_token)
        
        try:
            # Load point cloud data
            points = converter.load_sweeps(db, sample)
            
            # Create range view images
            range_image = converter.create_range_image(points, aggregation="closest")
            intensity_image = converter.create_intensity_image(points, aggregation="mean")
            
            # Save results
            base_filename = f"{sample_token}_{i:06d}"
            
            if args.save_images:
                # Save range image to its own subfolder
                save_range_view_image(
                    range_image, 
                    str(parent_ensured_path(args.output_dir / "images" / "range" / f"{base_filename}_range.png")),
                    title=f"Range View - {sample_token}",
                    colormap="plasma",
                    fov_azi_deg=tuple(args.fov_azi_deg)
                )
                
                # Save intensity image to its own subfolder
                save_range_view_image(
                    intensity_image,
                    str(parent_ensured_path(args.output_dir / "images" / "intensity" / f"{base_filename}_intensity.png")), 
                    title=f"Intensity View - {sample_token}",
                    colormap="gray",
                    fov_azi_deg=tuple(args.fov_azi_deg)
                )
            
            if args.save_data:
                # Save data as compressed numpy file
                metadata = {
                    'sample_token': sample_token,
                    'n_splits_azi': args.n_splits_azi,
                    'n_splits_pol': args.n_splits_pol,
                    'fov_azi_deg': args.fov_azi_deg,
                    'fov_pol_deg': args.fov_pol_deg,
                    'max_distance': args.max_distance,
                    'num_sweeps': args.num_sweeps,
                }
                
                save_range_view_data(
                    range_image,
                    intensity_image,
                    str(parent_ensured_path(args.output_dir / "data" / f"{base_filename}.npz")),
                    metadata
                )
                
        except Exception as e:
            print(f"Error processing sample {sample_token}: {e}")
            continue
    
    print(f"Completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()