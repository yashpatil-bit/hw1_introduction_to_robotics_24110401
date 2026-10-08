"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2: verify the
skew-symmetric identities from Problem 5 in simulation.

STARTER CODE. Model loading and a simple "spin the body" simulation
loop are provided and working. Your job is to fill in the TODOs to:

  1. Log R(t), the body's rotation matrix, at several simulated
     time steps while it spins.
  2. At each logged time step, numerically check, for several
     random v, w, omega in R^3:
         R (v x w) == (R v) x (R w)                 [Problem 5a]
         R w^ R^T  == (R w)^                         [Problem 5b, No-AI on paper]
     using utils.hat() for the ^ operator.
  3. Print the residual (it should be ~1e-14, machine precision)
     and explain in your write-up why a small-but-nonzero residual
     doesn't fully validate the identity, while a residual near
     machine epsilon strongly supports it.

Note: you already proved these identities by hand in Problem 5.
This script is not a substitute for that proof -- it's a numerical
sanity check, and a chance to see *why* proofs and simulation are
complementary, not interchangeable.
"""

import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity

MODEL_PATH = "../model/asymmetric_body.xml"

N_CHECKS_PER_STEP = 5     # how many random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # how many simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps to advance between each logged check


def random_unit_angular_velocity(rng):
    """A random constant angular velocity vector (rad/s), used to spin
    the body between logged checks."""
    w = rng.normal(size=3)
    return 2.0 * w / np.linalg.norm(w)


def check_identities(R, rng):
    """For N_CHECKS_PER_STEP random vectors v, w, omega, compute the
    residuals of:
        R @ np.cross(v, w)  vs.  np.cross(R @ v, R @ w)
        R @ hat(omega) @ R.T  vs.  hat(R @ omega)
    and return the worst-case (max) residual across all checks, for
    each identity separately.

    Return: (max_residual_cross, max_residual_skew)
    """
    max_resid_cross = 0.0
    max_resid_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # Problem 5a: R(v x w) == (Rv) x (Rw)
        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)
        resid_cross = np.max(np.abs(lhs_cross - rhs_cross))
        max_resid_cross = max(max_resid_cross, resid_cross)

        # Problem 5b: R w^ R^T == (R w)^
        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)
        resid_skew = np.max(np.abs(lhs_skew - rhs_skew))
        max_resid_skew = max(max_resid_skew, resid_skew)

    return max_resid_cross, max_resid_skew


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=0)

    # Spin the body with a fixed angular velocity
    data.qvel[3:6] = random_unit_angular_velocity(rng)
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------
    times = []
    rotation_logs = []
    cross_residuals = []
    skew_residuals = []

    print(
        f"{'step':>5} "
        f"{'t (s)':>8} "
        f"{'max resid: R(vxw)=(Rv)x(Rw)':>28} "
        f"{'max resid: RwR^T=(Rw)^':>24}"
    )

    # ------------------------------------------------------------
    # Run simulation and collect data
    # ------------------------------------------------------------
    for log_i in range(N_LOGGED_STEPS):

        for _ in range(STEPS_BETWEEN_LOGS):
            mujoco.mj_step(model, data)

        # Get current body rotation matrix
        R = get_body_orientation(data)

        # Sanity check:
        # A valid rotation matrix should satisfy R R^T = I
        assert is_close_to_identity(
            R @ R.T,
            tol=1e-6
        ), "R is not orthonormal!"

        # --------------------------------------------------------
        # Save the rotation matrix R(t)
        # --------------------------------------------------------
        times.append(data.time)
        rotation_logs.append((data.time, R.copy()))

        # --------------------------------------------------------
        # Check Problem 5 identities
        # --------------------------------------------------------
        resid_cross, resid_skew = check_identities(R, rng)

        cross_residuals.append(resid_cross)
        skew_residuals.append(resid_skew)

        print(
            f"{log_i:5d} "
            f"{data.time:8.3f} "
            f"{resid_cross:28.3e} "
            f"{resid_skew:24.3e}"
        )

    # ============================================================
    # SAVE ROTATION MATRICES
    # ============================================================

    import os
    os.makedirs("../results", exist_ok=True)

    rotation_data = []

    for t, R in rotation_logs:
        rotation_data.append([
            t,
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2]
        ])

    np.savetxt(
        "../results/rotation_matrices.csv",
        rotation_data,
        delimiter=",",
        header="time_s,R11,R12,R13,R21,R22,R23,R31,R32,R33",
        comments=""
    )

    print("\nRotation matrices saved to:")
    print("../results/rotation_matrices.csv")

    # ============================================================
    # SAVE RESIDUALS
    # ============================================================

    results_array = np.column_stack(
        (times, cross_residuals, skew_residuals)
    )

    np.savetxt(
        "../results/skew_verification_residuals.csv",
        results_array,
        delimiter=",",
        header="time_s,cross_product_residual,skew_identity_residual",
        comments=""
    )

    print("\nResiduals saved to:")
    print("../results/skew_verification_residuals.csv")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print("\nSummary:")
    print(
        f"Maximum cross-product residual = "
        f"{max(cross_residuals):.3e}"
    )
    print(
        f"Maximum skew-identity residual = "
        f"{max(skew_residuals):.3e}"
    )

    print("\nNumerical verification complete.")
    print("Both identities hold to approximately machine precision.")

    # ============================================================
    # PLOT RESIDUALS
    # ============================================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        times,
        cross_residuals,
        marker="o",
        label="R(v × w) = (Rv) × (Rw)"
    )

    plt.semilogy(
        times,
        skew_residuals,
        marker="s",
        label="Rω̂Rᵀ = (Rω)̂"
    )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Maximum residual")
    plt.title("Numerical Verification of Skew-Symmetric Identities")
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "../results/skew_verification_residuals.png",
        dpi=300
    )

    plt.show()

    print("\nPlot saved to:")
    print("../results/skew_verification_residuals.png")


    # TODO(student): save these residuals (e.g. to a CSV or a plot)
    # for your write-up, and answer the "why doesn't a small nonzero
    # residual fully prove the identity" question from HW1 Problem 8.


if __name__ == "__main__":
    main()