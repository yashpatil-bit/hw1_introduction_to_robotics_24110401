"""
01_rotation_current_frame.py
HW1 Part 2, Task 1: Rotation Order
Experiment 1 - Rotations about the CURRENT/BODY frame

Sequence:
    1. Rotate +90° about current z-axis
    2. Rotate +90° about current x-axis

For current-frame rotations:
    R_new = R_old @ R_step
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import (
    Rx,
    Ry,
    Rz,
    ELEMENTARY_ROTATIONS,
    set_body_orientation,
    hat,
    vee,
)

MODEL_PATH = "../model/asymmetric_body.xml"


# ===============================================================
# CURRENT FRAME ROTATION SEQUENCE
# ===============================================================
rotation_sequence = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
]


def compose_sequence(sequence):
    """
    Compose rotations about the CURRENT/BODY frame.

    Current-frame rotations are applied by right multiplication:

        R_new = R_old @ R_step
    """

    R = np.eye(3)

    for axis, angle in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        R = R @ R_step

    return R


def rotation_log(R):
    """
    Convert a rotation matrix into a rotation vector
    using the matrix logarithm.
    """

    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    w_hat = (R - R.T) / (2 * np.sin(theta))

    return vee(w_hat) * theta


def rotation_exp(w_vec):
    """
    Convert a rotation vector into a rotation matrix
    using Rodrigues' formula.
    """

    theta = np.linalg.norm(w_vec)

    if theta < 1e-9:
        return np.eye(3)

    w_hat = hat(w_vec / theta)

    return (
        np.eye(3)
        + np.sin(theta) * w_hat
        + (1 - np.cos(theta)) * (w_hat @ w_hat)
    )


def main():

    print("==============================================")
    print(" CURRENT FRAME ROTATION EXPERIMENT")
    print("==============================================")
    print(">>> Running current-frame rotation script <<<")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("Viewer open.")
        print("Rotation sequence:")
        print("1. +90° about current z-axis")
        print("2. +90° about current x-axis")

        # Start from identity orientation
        R_current = np.eye(3)

        set_body_orientation(data, R_current)
        mujoco.mj_forward(model, data)
        viewer.sync()

        print("\nModel loaded at starting orientation.")
        print("Position your camera in the viewer.")
        print("Starting in 15 seconds...\n")

        # Countdown for recording
        for remaining in range(15, 0, -1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1)

        print("Starting rotation now!\n")

        steps_per_rotation = 60

        # ===========================================================
        # APPLY EACH ROTATION
        # ===========================================================

        for axis, angle in rotation_sequence:

            R_step_full = ELEMENTARY_ROTATIONS[axis](angle)

            for i in range(1, steps_per_rotation + 1):

                frac = i / steps_per_rotation

                # Interpolate smoothly from 0° to the desired angle
                R_step_partial = rotation_exp(
                    rotation_log(R_step_full) * frac
                )

                # CURRENT FRAME:
                # Rotate about the body's moving axis
                R_display = R_current @ R_step_partial

                set_body_orientation(data, R_display)

                mujoco.mj_forward(model, data)
                viewer.sync()

                time.sleep(1 / 60)

            # Commit completed rotation
            R_current = R_current @ R_step_full

        # ===========================================================
        # FINAL RESULT
        # ===========================================================

        print("\nFinal orientation:")
        print(R_current)

        # Check result independently
        R_check = compose_sequence(rotation_sequence)

        assert np.allclose(
            R_current,
            R_check,
            atol=1e-6
        ), "Mismatch between animation and analytical result!"

        print("\nCurrent-frame experiment completed successfully.")

        # Keep viewer open
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()