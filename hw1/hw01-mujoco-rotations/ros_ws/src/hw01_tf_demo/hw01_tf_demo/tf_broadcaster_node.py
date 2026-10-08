"""
tf_broadcaster_node.py -- HW1 Part 2, Task 3 (optional/bonus).

Broadcasts:
    1. space_frame: fixed at the origin.
    2. body_frame: orientation changes using the same elemental
       rotation sequence as Task 1.

The ROS parameter:
    compose_frame = "current" or "fixed"

controls how each new rotation is composed:

    Current/body frame:
        R_new = R_old @ R_step

    Fixed/space frame:
        R_new = R_step @ R_old

Change the parameter live with:

    ros2 param set /hw01_tf_broadcaster compose_frame fixed

or:

    ros2 param set /hw01_tf_broadcaster compose_frame current
"""

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# ================================================================
# Elementary rotation matrices
# ================================================================

def Rx(t):
    """Rotation about x-axis."""
    c, s = np.cos(t), np.sin(t)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def Ry(t):
    """Rotation about y-axis."""
    c, s = np.cos(t), np.sin(t)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def Rz(t):
    """Rotation about z-axis."""
    c, s = np.cos(t), np.sin(t)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


# ================================================================
# Rotation sequence
# Same axis/angle sequence as Task 1
# ================================================================

ELEMENTARY_ROTATIONS = {
    "x": Rx,
    "y": Ry,
    "z": Rz
}

STEP_SEQUENCE = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
    ("y", np.deg2rad(60)),
]


# ================================================================
# Rotation matrix -> ROS quaternion
# ROS uses quaternion order: x, y, z, w
# ================================================================

def R_to_quat_xyzw(R):
    """
    Convert a 3x3 rotation matrix to a ROS-convention quaternion.

    Returns:
        [x, y, z, w]
    """

    tr = np.trace(R)

    if tr > 0:

        S = np.sqrt(tr + 1.0) * 2

        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

        S = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2

        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S

    elif R[1, 1] > R[2, 2]:

        S = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2

        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S

    else:

        S = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2

        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([x, y, z, w])

    # Normalize to avoid numerical drift
    return q / np.linalg.norm(q)


# ================================================================
# ROS2 TF Broadcaster Node
# ================================================================

class Hw01TfBroadcaster(Node):

    def __init__(self):

        super().__init__("hw01_tf_broadcaster")

        # --------------------------------------------------------
        # ROS parameters
        # --------------------------------------------------------

        self.declare_parameter(
            "compose_frame",
            "current"
        )

        self.declare_parameter(
            "step_period",
            1.0
        )

        # --------------------------------------------------------
        # TF broadcaster
        # --------------------------------------------------------

        self.tf_broadcaster = TransformBroadcaster(self)

        # --------------------------------------------------------
        # Initial body orientation
        # Identity matrix = no rotation
        # --------------------------------------------------------

        self.R_body = np.eye(3)

        # Which rotation step comes next
        self.step_index = 0

        # --------------------------------------------------------
        # Timer
        # --------------------------------------------------------

        step_period = self.get_parameter(
            "step_period"
        ).value

        self.timer = self.create_timer(
            step_period,
            self.on_timer
        )

        self.get_logger().info(
            "hw01_tf_broadcaster started."
        )

        self.get_logger().info(
            "Use:"
        )

        self.get_logger().info(
            "ros2 param set "
            "/hw01_tf_broadcaster "
            "compose_frame fixed"
        )

        self.get_logger().info(
            "or:"
        )

        self.get_logger().info(
            "ros2 param set "
            "/hw01_tf_broadcaster "
            "compose_frame current"
        )


    # ============================================================
    # Timer callback
    # ============================================================

    def on_timer(self):

        # --------------------------------------------------------
        # Broadcast fixed space frame
        # --------------------------------------------------------

        self.broadcast_frame(
            "world",
            "space_frame",
            np.eye(3)
        )

        # --------------------------------------------------------
        # Select next elemental rotation
        # --------------------------------------------------------

        axis, angle = STEP_SEQUENCE[
            self.step_index % len(STEP_SEQUENCE)
        ]

        R_step = ELEMENTARY_ROTATIONS[axis](angle)

        # --------------------------------------------------------
        # Read current composition mode
        # --------------------------------------------------------

        frame = self.get_parameter(
            "compose_frame"
        ).value

        # --------------------------------------------------------
        # Compose rotation
        #
        # CURRENT / BODY FRAME:
        #     R_new = R_old @ R_step
        #
        # FIXED / SPACE FRAME:
        #     R_new = R_step @ R_old
        # --------------------------------------------------------

        if frame == "current":

            self.R_body = (
                self.R_body @ R_step
            )

        elif frame == "fixed":

            self.R_body = (
                R_step @ self.R_body
            )

        else:

            self.get_logger().warn(
                f"Unknown compose_frame '{frame}'. "
                "Use 'current' or 'fixed'. "
                "Keeping previous orientation."
            )

        # --------------------------------------------------------
        # Move to next rotation in sequence
        # --------------------------------------------------------

        self.step_index += 1

        # --------------------------------------------------------
        # Broadcast body frame
        # --------------------------------------------------------

        self.broadcast_frame(
            "world",
            "body_frame",
            self.R_body
        )


    # ============================================================
    # Broadcast one TF transform
    # ============================================================

    def broadcast_frame(self, parent, child, R):

        t = TransformStamped()

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        t.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # --------------------------------------------------------
        # Parent and child frame names
        # --------------------------------------------------------

        t.header.frame_id = parent
        t.child_frame_id = child

        # --------------------------------------------------------
        # Translation
        #
        # space_frame -> origin
        # body_frame  -> placed 1 m above origin
        # --------------------------------------------------------

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0

        if child == "body_frame":
            t.transform.translation.z = 1.0
        else:
            t.transform.translation.z = 0.0

        # --------------------------------------------------------
        # Rotation
        # --------------------------------------------------------

        qx, qy, qz, qw = R_to_quat_xyzw(R)

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # --------------------------------------------------------
        # Send TF
        # --------------------------------------------------------

        self.tf_broadcaster.sendTransform(t)


# ================================================================
# Main
# ================================================================

def main(args=None):

    rclpy.init(args=args)

    node = Hw01TfBroadcaster()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


# ================================================================
# Program entry point
# ================================================================

if __name__ == "__main__":
    main()