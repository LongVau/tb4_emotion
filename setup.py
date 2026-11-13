from glob import glob
import os
from setuptools import find_packages, setup

package_name = "tb4_emotion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "stream_cam.py"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "models"), glob("models/*.onnx")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sumika",
    maintainer_email="sumika@todo.todo",
    description="Week 8: TurtleBot 4 Facial Emotion Recognition and Control",
    license="MIT",
    entry_points={
        "console_scripts": [
            "emotion_recognition_node = tb4_emotion.emotion_recognition_node:main",
            "emotion_control_node = tb4_emotion.emotion_control_node:main",
        ],
    },
)
