import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm  # 添加进度条支持

# 设备选择（CUDA 加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 定义 MLP 网络
class DarkChannelNet(nn.Module):
    def __init__(self):
        super(DarkChannelNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 输入维度修改为 2
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 限制输出在 [0,1] 之间
        return x

# 计算对比度
def compute_contrast(img):
    return np.std(img)

# 计算暗通道图
def dark_channel(img, size=7):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

# 估计大气光 A
def estimate_atmospheric_light(img, dark_channel):
    num_pixels = img.shape[0] * img.shape[1] // 1000
    dark_vec = dark_channel.ravel()
    img_vec = img.reshape(-1, 3)
    indices = np.argsort(dark_vec)[-num_pixels:]
    A = np.mean(img_vec[indices], axis=0)
    return A

# 计算透射率
def transmission_map(img, A, omega=0.95, size=7):
    normalized_img = img.astype(np.float32) / A
    dark = dark_channel(normalized_img, size)
    T = 1 - omega * dark
    return np.clip(T, 0.1, 1)

# 恢复去雾图像
def recover_image(img, T, A, t0=0.1):
    T = np.expand_dims(np.maximum(T, t0), axis=2)
    J = (img.astype(np.float32) - A) / T + A
    return np.clip(J, 0, 255).astype(np.uint8)

# 设定更宽的可调范围
A_min, A_max = 0, 255  # 大气光最小值和最大值
T_min, T_max = 0.2, 1.2  # 透射率最小值和最大值

def process_frame(frame, net, omega=0.85):
    contrast = compute_contrast(frame)

    # 通过模型计算 A 和 T 的调整量
    input_tensor = torch.tensor([contrast, 1.0], dtype=torch.float32, device=device, requires_grad=True)  # 调整输入为 2 维
    dark_param = net(input_tensor)

    A_adjust, T_adjust = dark_param.item(), dark_param.item() * 0.1  # 使用模型的输出调整 A 和 T

    patch_size = max(3, min(9, int(dark_param.item() * 20)))
    dark_channel_img = dark_channel(frame, patch_size)
    A = estimate_atmospheric_light(frame, dark_channel_img) + A_adjust
    A = np.clip(A, A_min, A_max)  # 限制 A 在合理范围内

    T = transmission_map(frame, A, omega, patch_size) + T_adjust
    T = np.clip(T, T_min, T_max)  # 限制 T 在合理范围内

    recovered_img = recover_image(frame, T, A)

    return recovered_img, A, T

def test_dynamic_adjustment(video_path, model_path):
    # 加载训练好的模型
    net = DarkChannelNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理每一帧图像
            recovered_img, A, T = process_frame(frame, net)

            # 显示原始图像和恢复后的图像
            cv2.imshow("Original", frame)
            cv2.imshow("Recovered Image", recovered_img)

            # 输出动态调整的 A 和 T
            print(f"Adjusted A: {A}, Adjusted T: {T}")

            # 修改 waitKey 参数，让视频自动播放
            key = cv2.waitKey(10)  # 10 毫秒

            if key == 27:  # ESC 退出
                break

            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

# 测试模型
video_path = "/home/wsy/Documents/VedioData/output_tr_smoked.mp4"  # 修改为你的视频路径
model_path = "/home/wsy/python_Projects/test3/dark_channel_model.pth"  # 修改为你训练模型的路径
test_dynamic_adjustment(video_path, model_path)
