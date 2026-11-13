"""Launch file starting emotion_recognition_node and emotion_control_node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    camera_source = LaunchConfiguration("camera_source")
    show_window = LaunchConfiguration("show_window")
    face_loss_timeout = LaunchConfiguration("face_loss_timeout")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_source",
                default_value="auto",
                description="Webcam index (0), 'auto' (tries /dev/video0 then Windows WSL2 HTTP stream), or HTTP URL",
            ),
            DeclareLaunchArgument(
                "show_window",
                default_value="true",
                description="Show OpenCV emotion recognition window",
            ),
            DeclareLaunchArgument(
                "face_loss_timeout",
                default_value="2.0",
                description="Seconds to keep moving after face is lost before stopping",
            ),
            Node(
                package="tb4_emotion",
                executable="emotion_recognition_node",
                name="emotion_recognition_node",
                output="screen",
                parameters=[
                    {
                        "camera_source": camera_source,
                        "show_window": show_window,
                    }
                ],
            ),
            Node(
                package="tb4_emotion",
                executable="emotion_control_node",
                name="emotion_control_node",
                output="screen",
                parameters=[
                    {
                        "face_loss_timeout": face_loss_timeout,
                    }
                ],
            ),
        ]
    )
