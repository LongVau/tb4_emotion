# Package ROS 2: `tb4_emotion`
**Bài thực hành Tuần 8 – Nhận diện biểu cảm khuôn mặt và điều khiển TurtleBot 4**

Package ROS 2 Humble độc lập nhận diện 6 biểu cảm khuôn mặt thời gian thực qua webcam hoặc luồng HTTP MJPEG từ Windows vào WSL2 (`stream_cam.py`), sử dụng mô hình Deep Learning pretrained (`emotion-ferplus-8.onnx` & `face_detection_yunet_2023mar.onnx`) kết hợp OpenCV DNN và điều khiển TurtleBot 4 qua topic `/cmd_vel` với cơ chế chạy liên tục và deadzone 2 giây khi mất khuôn mặt.

---

## 1. Bảng ánh xạ biểu cảm & Hành động của TurtleBot (`tuan_8.png`)

| Biểu cảm | Hành động của TurtleBot | Vận tốc `/cmd_vel` (`linear.x`, `angular.z`) |
| :--- | :--- | :--- |
| **Happy** (Vui vẻ) | Di chuyển theo quỹ đạo cong sang trái | `(0.18 m/s, +0.45 rad/s)` liên tục |
| **Sad** (Buồn) | Di chuyển theo quỹ đạo cong sang phải | `(0.18 m/s, -0.45 rad/s)` liên tục |
| **Neutral** (Bình thường) | Dừng tại chỗ | `(0.0 m/s, 0.0 rad/s)` |
| **Angry** (Tức giận) | Lùi nhẹ về phía sau | `(-0.12 m/s, 0.0 rad/s)` liên tục |
| **Surprise** (Ngạc nhiên) | Xoay một vòng tại chỗ | `(0.0 m/s, +1.0 rad/s)` đủ `2π` rad (~6.28s) rồi dừng |
| **Fear** (Sợ hãi) | Tiến nhẹ về phía trước | `(+0.12 m/s, 0.0 rad/s)` liên tục |
| **None** (Mất mặt) | Giữ quỹ đạo trong **2.0s deadzone**, sau đó dừng | Quá `2.0s` không thấy mặt -> `(0.0, 0.0)` |

---

## 2. Biên dịch & Kiểm thử (Build & Test)

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select tb4_emotion
source install/setup.bash
```

Chạy unit test:
```bash
PYTHONPATH=~/ros2_ws/src/tb4_emotion python3 -m pytest -p no:anyio ~/ros2_ws/src/tb4_emotion/test/ -v
```

---

## 3. Hướng dẫn Stream Camera từ Windows vào WSL2 & Chạy chương trình

### Bước 1: Chạy `stream_cam.py` trên Windows (nếu dùng Webcam laptop qua WSL2)
Mở PowerShell / CMD trên Windows và chạy file đã được copy sẵn tại `C:\Users\sumika\stream_cam.py` (hoặc qua đường dẫn WSL `\\wsl.localhost\Ubuntu-22.04\home\sumika\ros2_ws\src\tb4_emotion\stream_cam.py`):
```powershell
python C:\Users\sumika\stream_cam.py
```
Server MJPEG sẽ phát tại cổng `5005` (`/video`).

### Bước 2: Mở mô phỏng TurtleBot 4 trên Gazebo (Terminal WSL2 #1)
Mở một terminal WSL2 riêng để khởi chạy mô phỏng TurtleBot 4:
```bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
```
*(Nếu muốn mở kèm RViz hoặc đổi world/chế độ chạy ngay khi khởi động có thể dùng lệnh đầy đủ):*
```bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py world:=depot model:=standard rviz:=true
```
> **Lưu ý:** Sau khi cửa sổ Ignition Gazebo hiện lên, nếu mô phỏng đang tạm dừng thì bấm nút **Play (▶)** góc dưới bên trái cửa sổ Gazebo để đồng hồ thời gian thực và robot bắt đầu nhận lệnh `/cmd_vel`.

### Bước 3: Khởi chạy hệ thống nhận diện biểu cảm (Terminal WSL2 #2)
Chế độ mặc định `camera_source:=auto` sẽ tự động thử `/dev/video0` trước, nếu không có sẽ tự động dò IP Windows Host (`ip route show default`) và kết nối vào `http://<WINDOWS_IP>:5005/video`:
```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tb4_emotion emotion_control.launch.py
```

Hoặc truyền trực tiếp URL của Windows Host (tương tự `tb4_gesture`):
```bash
ros2 launch tb4_emotion emotion_control.launch.py camera_source:=http://172.23.64.1:5005/video
```
