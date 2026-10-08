"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation, hat, vee

MODEL_PATH = "../model/asymmetric_body.xml"

# ---------------------------------------------------------------
# Edit this to switch between the "current frame" and "fixed frame"
# experiments. Same (axis, angle) pairs, different frame choice.
# ---------------------------------------------------------------
rotation_sequence = [
    ("z", np.deg2rad(90), "current"),
    ("x", np.deg2rad(90), "current"),
]

# Uncomment this one and comment the above to run the fixed-frame version:
# rotation_sequence = [
#     ("z", np.deg2rad(90), "fixed"),
#     ("x", np.deg2rad(90), "fixed"),
# ]


def compose_sequence(sequence):
    """Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.

    - "current" frame -> rotate about the body's own (moving) axes
      -> post-multiply (right-multiply): R = R_old @ R_step
    - "fixed" frame -> rotate about the fixed space axes
      -> pre-multiply (left-multiply): R = R_step @ R_old
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        if frame == "current":
            R = R @ R_step
        elif frame == "fixed":
            R = R_step @ R
        else:
            raise ValueError(f"Unknown frame: {frame!r}")
    return R


def rotation_log(R):
    """Matrix logarithm: rotation matrix -> rotation vector (axis*angle)."""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    w_hat = (R - R.T) / (2 * np.sin(theta))
    return vee(w_hat) * theta


def rotation_exp(w_vec):
    """Matrix exponential (Rodrigues' formula): rotation vector -> matrix."""
    theta = np.linalg.norm(w_vec)
    if theta < 1e-9:
        return np.eye(3)
    w_hat = hat(w_vec / theta)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)


def main():
    print(">>> RUNNING UPDATED SCRIPT WITH COUNTDOWN <<<", flush=True)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open.")
        print(f"Applied sequence: {rotation_sequence}")

        R_current = np.eye(3)
        set_body_orientation(data, R_current)
        mujoco.mj_forward(model, data)
        viewer.sync()

        print("\nModel is loaded at its starting orientation.")
        print("Position your camera in the viewer window now (drag to orbit, scroll to zoom).")
        print("Starting in 15 seconds -- use this time to start your screen recording.\n")
        for remaining in range(15, 0, -1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1)
        print("Starting rotation now!\n")

        steps_per_rotation = 60  # ~1 second per elemental rotation at 60Hz
        for axis, angle, frame in rotation_sequence:
            R_step_full = ELEMENTARY_ROTATIONS[axis](angle)
            for i in range(1, steps_per_rotation + 1):
                frac = i / steps_per_rotation
                R_step_partial = rotation_exp(rotation_log(R_step_full) * frac)
                if frame == "current":
                    R_display = R_current @ R_step_partial
                else:  # fixed
                    R_display = R_step_partial @ R_current
                set_body_orientation(data, R_display)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1 / 60)
            # commit the full step before starting the next one
            if frame == "current":
                R_current = R_current @ R_step_full
            else:
                R_current = R_step_full @ R_current

        print(f"Final orientation R =\n{R_current}")

        # confirm compose_sequence agrees with the animated result
        R_check = compose_sequence(rotation_sequence)
        assert np.allclose(R_current, R_check, atol=1e-6), "Mismatch between animated and compose_sequence result!"

        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()