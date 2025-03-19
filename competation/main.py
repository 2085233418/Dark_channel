import cv2
import numpy as np

def dark_channel(img, size=7):
    """计算暗通道"""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark_channel):
    """估计大气光A"""
    num_pixels = img.shape[0] * img.shape[1] // 1000
    dark_vec = dark_channel.ravel()
    img_vec = img.reshape(-1, 3)
    indices = np.argsort(dark_vec)[-num_pixels:]
    A = np.mean(img_vec[indices], axis=0)
    return A

def transmission_map(img, A, omega=0.95, size=7):
    """计算透射率"""
    normalized_img = img.astype(np.float32) / A
    dark = dark_channel(normalized_img, size)
    T = 1 - omega * dark
    return np.clip(T, 0.1, 1)

def recover_image(img, T, A, t0=0.1):
    """恢复去雾图像"""
    T = np.expand_dims(np.maximum(T, t0), axis=2)
    J = (img.astype(np.float32) - A) / T + A
    return np.clip(J, 0, 255).astype(np.uint8)

# 读取视频
video_path = "/home/wsy/Documents/VedioData/output_tr_smoked3.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 保存去雾后的视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_dehazed.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 结束视频读取

    # 计算暗通道
    dark = dark_channel(frame, 15)

    # 估计大气光
    A = estimate_atmospheric_light(frame, dark)

    # 计算透射率
    T = transmission_map(frame, A, 0.95, 15)

    # 去雾处理
    dehazed_frame = recover_image(frame, T, A)

    # 显示原始和去雾后的视频
    cv2.imshow("Original Video", frame)
    cv2.imshow("Dehazed Video", dehazed_frame)

    # 保存去雾后帧
    out.write(dehazed_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
