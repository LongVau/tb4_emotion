#!/usr/bin/env python3
"""ROS 2 Node: emotion_control_node.

Subscribes to /detected_emotion (std_msgs/msg/String) and drives TurtleBot 4
continuously via /cmd_vel (geometry_msgs/msg/Twist) according to the 6 facial
emotions in tuan_8.png, with a 2.0-second face-loss deadzone before stopping.
"""

import math
import time
from typing import Optional, Tuple

VALID_EMOTIONS = ("Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear")


class EmotionController:
    """Continuous emotion-to-velocity controller with 2.0s face-loss deadzone."""

    def __init__(
        self,
        face_loss_timeout: float = 2.0,
        linear_curve: float = 0.18,
        angular_curve: float = 0.45,
        linear_slow: float = 0.12,
        spin_angular: float = 1.0,
    ) -> None:
        self.face_loss_timeout = face_loss_timeout
        self.linear_curve = linear_curve
        self.angular_curve = angular_curve
        self.linear_slow = linear_slow
        self.spin_angular = spin_angular

        self.active_emotion: str = "None"
        self.last_face_time: Optional[float] = None
        self.spin_start_time: Optional[float] = None
        self.spin_completed: bool = False

    def on_emotion(self, emotion: str, now: float) -> None:
        label = emotion.strip()
        if label not in VALID_EMOTIONS:
            return

        self.last_face_time = now
        if label != self.active_emotion:
            self.active_emotion = label
            if label == "Surprise":
                self.spin_start_time = now
                self.spin_completed = False
            else:
                self.spin_start_time = None
                self.spin_completed = False

    def step(self, now: float) -> Tuple[float, float]:
        if (
            self.last_face_time is None
            or (now - self.last_face_time) > self.face_loss_timeout
        ):
            self.active_emotion = "None"
            self.spin_start_time = None
            self.spin_completed = False
            return 0.0, 0.0

        emo = self.active_emotion
        if emo == "Happy":
            return self.linear_curve, self.angular_curve
        if emo == "Sad":
            return self.linear_curve, -self.angular_curve
        if emo == "Neutral":
            return 0.0, 0.0
        if emo == "Angry":
            return -self.linear_slow, 0.0
        if emo == "Fear":
            return self.linear_slow, 0.0
        if emo == "Surprise":
            if self.spin_completed or self.spin_start_time is None:
                return 0.0, 0.0
            duration = (2.0 * math.pi) / max(1e-6, abs(self.spin_angular))
            if (now - self.spin_start_time) >= duration:
                self.spin_completed = True
                return 0.0, 0.0
            return 0.0, self.spin_angular

        return 0.0, 0.0


def main(args=None) -> None:
    from geometry_msgs.msg import Twist
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class EmotionControlNode(Node):
        def __init__(self) -> None:
            super().__init__("emotion_control_node")
            self.declare_parameter("face_loss_timeout", 2.0)
            timeout = float(self.get_parameter("face_loss_timeout").value)
            self.controller = EmotionController(face_loss_timeout=timeout)
            self.last_logged_state = ("None", False)

            self.sub = self.create_subscription(
                String, "/detected_emotion", self._on_msg, 10
            )
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.timer = self.create_timer(0.05, self._tick)
            self.get_logger().info(
                f"emotion_control_node ready (face_loss_timeout={timeout:.1f}s)"
            )

        def _on_msg(self, msg: String) -> None:
            self.controller.on_emotion(msg.data, time.monotonic())

        def _tick(self) -> None:
            now = time.monotonic()
            lin, ang = self.controller.step(now)
            cur_state = (
                self.controller.active_emotion,
                self.controller.spin_completed,
            )
            if cur_state != self.last_logged_state:
                self.last_logged_state = cur_state
                self.get_logger().info(
                    f"Active emotion='{cur_state[0]}' -> cmd_vel=({lin:.2f}, {ang:.2f})"
                )

            cmd = Twist()
            cmd.linear.x = float(lin)
            cmd.angular.z = float(ang)
            self.pub.publish(cmd)

    rclpy.init(args=args)
    node = EmotionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.pub.publish(stop)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
