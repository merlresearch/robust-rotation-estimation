# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import csv
import math
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robust_estimation import RobustRotationEstimator


def load(sequence_path: str):
    """
    Load a sequence of the BUSS dataset
    :param sequence_path: Path to the BUSS sequence
    :return: Dict
    """

    # Ground truth rotations
    rotations = []
    # Optical flows
    flows = []

    with open(os.path.join(sequence_path, "rotations.csv"), mode="r") as rotations_file:

        for i, line in enumerate(csv.reader(rotations_file)):
            if i == 0:
                continue

            # Ground truth rotations
            rotations.append(Rotation.from_quat([float(e) for e in line]))

            # Flows
            flow_folder = os.path.join(sequence_path, "flows_undistorted_4")
            flows.append(torch.load(os.path.join(flow_folder, f"{i - 1:06d}_4.pt")))

    return {
        "flows": flows,
        "rotations": rotations,
    }


def evaluate(buss_path: str):
    """
    Evaluate on BUSS dataset
    :param buss_path: Path to the BUSS dataset
    """
    rotation_estimator = RobustRotationEstimator(
        h=270, w=480, f=1655 / 4, bin_size=0.001, max_angle=0.07, spatial_step=15
    )

    errors = []

    for sequence in os.listdir(buss_path):

        sequence_path = os.path.join(buss_path, sequence)
        if os.path.isdir(sequence_path):
            print(f"Evaluating sequence {sequence}")
            data = load(sequence_path)

            assert len(data["flows"]) == len(
                data["rotations"]
            ), f"Expected {len(data['flows'])} == {len(data['rotations'])}"

            for i, flow in enumerate(data["flows"]):
                pred_rot_euler = rotation_estimator.estimate(flow)
                pred_rot_euler[-1] = -pred_rot_euler[-1]
                errors.append((Rotation.from_rotvec(pred_rot_euler).inv() * data["rotations"][i]).magnitude())

    errors = np.array(errors)
    print(f"Average error in deg: {errors.sum() / len(errors) / math.pi * 180}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buss_path", type=str, required=True, help="Path to the buss dataset")
    args = parser.parse_args()

    evaluate(args.buss_path)
