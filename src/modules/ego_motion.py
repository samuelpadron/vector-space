"""
Ego-motion estimation for SE(2) transforms.
Supports both nuScenes ego_pose/velocity and GPS/speedometer data.
Provides SE(2) transforms (Δx, Δy, Δyaw) for BEV feature warping.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SE2Transform:
    """2D rigid body transform in BEV grid coordinates."""
    dx: float  # grid units
    dy: float  # grid units
    dyaw: float  # radians


@dataclass
class EgoPose:
    """Ego vehicle pose: position and rotation."""
    x: float
    y: float
    z: float
    roll: float  # radians
    pitch: float  # radians
    yaw: float  # radians


class GPSToENUConverter:
    """Convert lat/lon to Earth-Centered, Earth-Fixed (ENU) local coordinates."""

    def __init__(self, reference_lat: float, reference_lon: float):
        """
        Args:
            reference_lat, reference_lon: Origin point for ENU frame (typically first frame).
        """
        self.ref_lat = np.radians(reference_lat)
        self.ref_lon = np.radians(reference_lon)

        # WGS84 constants
        self.a = 6378137.0  # Earth semi-major axis (meters)
        self.e2 = 0.00669438  # WGS84 eccentricity squared

    def to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to ENU (East, North) in meters.

        Args:
            lat, lon: Degrees

        Returns:
            (east, north) in meters relative to reference.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat_rad)**2)

        x = (N + 0) * np.cos(lat_rad) * (lon_rad - self.ref_lon)
        y = (N * (1 - self.e2) + 0) * (lat_rad - self.ref_lat)

        return float(x), float(y)


class EgoMotionEstimator:
    """Estimate ego-motion from both nuScenes and real-world (GPS/speedometer) data."""

    def __init__(self, grid_size_m: float = 51.2, grid_resolution: float = 0.8):
        """
        Args:
            grid_size_m: Total BEV grid extent (e.g., [-51.2, 51.2]m).
            grid_resolution: Meters per BEV cell (e.g., 0.8m).
        """
        self.grid_size_m = grid_size_m
        self.grid_resolution = grid_resolution
        self.gps_converter: Optional[GPSToENUConverter] = None

    def estimate_from_ego_pose(
        self,
        pose_prev: EgoPose,
        pose_curr: EgoPose,
    ) -> SE2Transform:
        """
        Estimate SE(2) from ego poses (nuScenes format).

        Args:
            pose_prev: EgoPose at t-1
            pose_curr: EgoPose at t

        Returns:
            SE2Transform with (dx, dy, dyaw) in grid units.
        """
        # Extract position delta in meters
        dx_m = pose_curr.x - pose_prev.x
        dy_m = pose_curr.y - pose_prev.y

        # Extract yaw delta from rotation (assuming z-axis rotation is yaw)
        dyaw = pose_curr.yaw - pose_prev.yaw

        # Normalize yaw to [-π, π]
        while dyaw > np.pi:
            dyaw -= 2 * np.pi
        while dyaw < -np.pi:
            dyaw += 2 * np.pi

        # Convert meters to grid units
        dx_grid = dx_m / self.grid_resolution
        dy_grid = dy_m / self.grid_resolution

        return SE2Transform(dx=dx_grid, dy=dy_grid, dyaw=dyaw)

    def estimate_from_gps(
        self,
        pos_prev: Tuple[float, float],
        pos_curr: Tuple[float, float],
        heading_curr: float,
        heading_prev: float,
    ) -> SE2Transform:
        """
        Estimate SE(2) from GPS coordinates (lat/lon) and heading.

        Args:
            pos_prev: (lat, lon) at t-1
            pos_curr: (lat, lon) at t
            heading_prev: Degrees at t-1
            heading_curr: Degrees at t

        Returns:
            SE2Transform with (dx, dy, dyaw) in grid units.
        """
        # Initialize ENU converter on first call
        if self.gps_converter is None:
            self.gps_converter = GPSToENUConverter(pos_prev[0], pos_prev[1])

        # Convert to ENU
        e_prev, n_prev = self.gps_converter.to_enu(pos_prev[0], pos_prev[1])
        e_curr, n_curr = self.gps_converter.to_enu(pos_curr[0], pos_curr[1])

        # Compute displacement
        de = e_curr - e_prev
        dn = n_curr - n_prev

        # Yaw from heading (unreliable at low speed; consider visual odometry fallback)
        dyaw = np.radians(heading_curr - heading_prev)

        # Normalize yaw
        while dyaw > np.pi:
            dyaw -= 2 * np.pi
        while dyaw < -np.pi:
            dyaw += 2 * np.pi

        # Convert meters to grid units
        dx_grid = de / self.grid_resolution
        dy_grid = dn / self.grid_resolution

        return SE2Transform(dx=dx_grid, dy=dy_grid, dyaw=dyaw)

    def se2_to_affine_matrix(self, se2: SE2Transform) -> torch.Tensor:
        """
        Convert SE(2) to 2×3 affine matrix for grid_sample.

        Args:
            se2: SE2Transform

        Returns:
            Tensor of shape [1, 2, 3] for use with grid_sample.
        """
        c, s = np.cos(se2.dyaw), np.sin(se2.dyaw)

        # Affine matrix: rotation + translation, normalized to [-1, 1] grid
        affine = torch.tensor([
            [c, -s, se2.dx / (self.grid_size_m / self.grid_resolution / 2)],
            [s,  c, se2.dy / (self.grid_size_m / self.grid_resolution / 2)],
        ], dtype=torch.float32).unsqueeze(0)

        return affine
