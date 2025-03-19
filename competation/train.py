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
        self.fc1 = nn.Linear(2, 16)
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

# 计算均匀度
def compute_uniformity(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-5))
    return entropy

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

# 处理单帧
'''def process_frame(frame, net, criterion, optimizer, feedback, omega=0.95):
    contrast = compute_contrast(frame)
    uniformity = compute_uniformity(frame)

    input_tensor = torch.tensor([contrast, uniformity], dtype=torch.float32, device=device, requires_grad=True)
    dark_param = net(input_tensor)

    if feedback is None:
        feedback = (0, 0)  # 默认不调整

    A_adjust, T_adjust = feedback
    patch_size = max(3, min(9, int(dark_param.item() * 20)))
    dark_channel_img = dark_channel(frame, patch_size)
    A = estimate_atmospheric_light(frame, dark_channel_img) + A_adjust
    T = transmission_map(frame, A, omega, patch_size) + T_adjust

    recovered_img = recover_image(frame, T, A)

    # 训练模型，即使用户不提供反馈，每 100 帧自动学习
    loss = criterion(dark_param, torch.tensor([feedback[0]], dtype=torch.float32, device=device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return recovered_img'
    '''
# 设定更宽的可调范围
A_min, A_max = 0, 255  # 大气光最小值和最大值
T_min, T_max = 0.2, 1.2  # 透射率最小值和最大值

def process_frame(frame, net, criterion, optimizer, feedback, omega=0.85):
    contrast = compute_contrast(frame)
    uniformity = compute_uniformity(frame)

    input_tensor = torch.tensor([contrast, uniformity], dtype=torch.float32, device=device, requires_grad=True)
    dark_param = net(input_tensor)

    if feedback is None:
        feedback = (0, 0)  # 默认不调整

    A_adjust, T_adjust = feedback
    patch_size = max(3, min(9, int(dark_param.item() * 20)))
    dark_channel_img = dark_channel(frame, patch_size)
    A = estimate_atmospheric_light(frame, dark_channel_img) + A_adjust
    A = np.clip(A, A_min, A_max)  # 限制 A 在合理范围内

    T = transmission_map(frame, A, omega, patch_size) + T_adjust
    T = np.clip(T, T_min, T_max)  # 限制 T 在合理范围内

    recovered_img = recover_image(frame, T, A)

    # 训练模型，即使用户不提供反馈，每 100 帧自动学习
    loss = criterion(dark_param, torch.tensor([feedback[0]], dtype=torch.float32, device=device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return recovered_img


# 训练模型
def train_model(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    net = DarkChannelNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    model_path = "/home/wsy/python_Projects/test3/dark_channel_model.pth"#the path to save model
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model.")

    feedback = None
    frame_counter = 0

    with tqdm(total=frame_count, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 每 100 帧重新加载一次模型，确保最新参数
            if frame_counter % 100 == 0 and os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path, map_location=device))
                print("Reloaded updated model.")

            recovered_img = process_frame(frame, net, criterion, optimizer, feedback)

            cv2.imshow("Original", frame)
            cv2.imshow("Recovered Image", recovered_img)

            # 修改 waitKey 参数，让视频自动播放
            key = cv2.waitKey(10)  # 10 毫秒

            # 监听用户调整参数
            if key == ord('1'):
                feedback = (0.05, 0)  # 增加大气光
            elif key == ord('4'):
                feedback = (0, 0.05)  # 增加透射率
            elif key == ord('3'):
                feedback = (-0.05, 0)  # 减少大气光
            elif key == ord('6'):
                feedback = (0, -0.05)  # 减少透射率
            elif key == 27:  # ESC 退出
                break

            frame_counter += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    torch.save(net.state_dict(), model_path,_use_new_zipfile_serialization=False)
    print(f"Model saved to {model_path}")

# 运行训练
train_model("/home/wsy/Documents/VedioData/output_rgb_smoked3.mp4")#change to your vedio data path
#output_rgb.mp4--done
#output_tr_smoked.mp4--done
#output_tr_smoked1.mp4--done
#output_tr_smoked2.mp4--done
#output_tr_smoked3.mp4--done
#output_tr.mp4--done
#output_rgb_smoked1.mp4--done
#output_rgb_smoked2.mp4--done