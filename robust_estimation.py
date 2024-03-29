# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import math

import numpy as np
from scipy import stats


class RobustRotationEstimator:
    def __init__(self, h: int, w: int, f: float, bin_size: float, max_angle: float, spatial_step: int):
        """
        Robust Rotation Estimator
        Args:
            h: Height of the flow field
            w: Width of the flow field
            f: Focal Length
            bin_size: Size of the sides of the 3D bins in the rotation space in radians
            max_angle: Rotations to search go from - max_angle to + max_angle
            spatial_step: Samples the flow 1 pixel every spatial_step pixels.
        """

        self.h = h
        self.w = w
        self.f = f
        self.bin_size = bin_size
        self.max_angle = max_angle
        self.spatial_step = spatial_step

        self.center = np.array([0.0, 0.0, 0.0])
        self.shift = 0
        self.n_points = 2

        # For parameters:
        # hist step: 0.001
        # hist max: 0.07
        # First bin: -0.0705 / -0.0695
        # Bin in the middle: -0.0005 / 0.0005
        # Last bin: .0695 / 0.0705

        self.middle_bins = np.arange(-self.max_angle, self.max_angle + self.bin_size, self.bin_size) + self.shift
        self.start_bins = self.middle_bins - self.bin_size / 2
        self.n_bins_per_dim = len(self.middle_bins)

        self.flow_locations = self._get_flow_vector_locations()

        # Precompute lines for rotation estimation
        self.lines = self._precompute_lines_vectorized()  # H/spatial_step, W/spatial_step, n_rot, 3

    def _get_flow_vector_locations(self):
        """
        Return an array representing the indices of a grid.
        :return: Array of indices np.array ((x, y), w, h)
        """

        c = np.arange(math.floor(-(self.h - 1) / 2), math.floor((self.h - 1) / 2) + 1)
        r = np.arange(math.floor(-(self.w - 1) / 2), math.floor((self.w - 1) / 2) + 1)
        out = np.zeros((2, self.h, self.w))
        out[0, :, :] = r
        out[1, :, :] = c[:, None]
        out = out[:, :: self.spatial_step, :: self.spatial_step]
        return out.astype(np.float64) + 0.5

    def _precompute_lines_vectorized(self):
        """
        Precompute the lines of compatible rotations
        :return: Ys, Xs, n_rotations, 3
        """

        # 2 times more points than bins, this is arbitrary
        n_rotations = self.n_bins_per_dim * self.n_points

        # Normal vectors to the planes
        eu = np.stack(
            (
                (self.flow_locations[0] * self.flow_locations[1]) / self.f,
                -(self.f + self.flow_locations[0] ** 2 / self.f),
                self.flow_locations[1],
            ),
            axis=-1,
        )

        ev = np.stack(
            (
                (self.f + self.flow_locations[1] ** 2 / self.f),
                -(self.flow_locations[0] * self.flow_locations[1]) / self.f,
                -self.flow_locations[0],
            ),
            axis=-1,
        )

        v_s = np.cross(eu, ev)

        v_s = v_s / np.linalg.norm(v_s, axis=-1)[:, :, None]

        points = np.linspace(
            v_s / v_s[:, :, 2:3] * self.middle_bins[0],
            v_s / v_s[:, :, 2:3] * self.middle_bins[-1],
            num=n_rotations,  # num=(n_rotations - 1)
        )

        points = np.transpose(points, (1, 2, 0, 3))

        return points

    def _rad_to_idx(self, x):
        """
        Angle in radian to index
        Args:
            x (torch.tensor): (..., 3)

        Returns: Idx


        """

        #  0.07      0.069
        #
        #  |----------|----------|
        #       <----   Need to floor, the prediction will then be the middle of the bin

        return np.floor((x + self.max_angle + self.bin_size / 2) / self.bin_size - self.shift).astype(np.int64)

    def _idx_to_rad(self, x):
        """
        Idx to angle in radian
        :param x: Idx
        :return:  Rad
        """
        return self.middle_bins[x]

    def estimate(self, flow):
        """

        :param flow: np.array Optical flow (h, w, 2)
        :return: Rotation
        """

        indices_x, indices_y = self.flow_locations

        u = flow[:: self.spatial_step, :: self.spatial_step, 0]
        v = flow[:: self.spatial_step, :: self.spatial_step, 1]

        a = (self.f**2 * v - u * indices_x * indices_y + v * indices_x**2) / (
            self.f**3 + self.f * indices_x**2 + self.f * indices_y**2
        )
        b = -(self.f**2 * u + u * indices_y**2 - v * indices_x * indices_y) / (
            self.f**3 + self.f * indices_x**2 + self.f * indices_y**2
        )

        point = np.stack([a, b, np.zeros_like(a)], axis=-1)  # (SH, SW, 3)
        point = point[:, :, None, :]  # (SH, SW, 1, 3)

        lines_c = self.lines + point  # (SH, SW, n_rot, 3)

        # Flatten spatial dimension
        votes = np.reshape(lines_c, (-1, 3))  # (N, 3)

        # Convert votes from radian to indices
        votes = self._rad_to_idx(votes)

        # Delete out of bounds indices
        indices_to_keep = ((votes >= 0) & (votes < self.n_bins_per_dim)).all(1)
        votes = votes[indices_to_keep]

        # Flatten the indices
        all_indexes = np.ravel_multi_index(votes.T, [self.n_bins_per_dim] * 3)

        # Find the mode of the list of votes
        mode, _ = stats.mode(all_indexes)

        # Unflatten the winning index
        indexes = np.array(np.unravel_index(mode, [self.n_bins_per_dim] * 3))

        # Winning index to radian
        pred_rot = self._idx_to_rad(indexes)

        return pred_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=str, required=True, help="Path to optical flow")
    parser.add_argument("--f", type=float, default=1655 / 4, help="Focal length")

    parser.add_argument("--spatial_step", type=int, default=15, help="Flow grid sample rate")
    parser.add_argument("--bin_size", type=float, default=0.001, help="Size of the rotation bins in radian")
    parser.add_argument(
        "--max_angle",
        type=float,
        default=0.07,
        help="Range of rotations to search for in radian. The rotation will be searched within [-max_angle, max_angle]",
    )
    args = parser.parse_args()

    flow = np.load(args.flow)

    # Frame size
    h, w, _ = flow.shape

    # Focal length
    f = 1655 / 4

    rotation_estimator = RobustRotationEstimator(h, w, args.f, args.bin_size, args.max_angle, args.spatial_step)

    rot_est = rotation_estimator.estimate(flow)

    out_string = f"Estimated rotation: {rot_est}"
    print(out_string)
