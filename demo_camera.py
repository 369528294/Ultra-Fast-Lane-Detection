"""
摄像头或视频实时车道线检测，在窗口中以绿点+红色中心线显示。

使用说明
--------
  --model   模型权重路径（默认：weights/tusimple_18.pth）
  --source  视频文件路径；不传则使用摄像头
  --seconds 仅在使用视频时有效：只播放前 N 秒（默认：10）
  --camera  使用摄像头时的设备号，0 为默认（默认：0）
  --no_cuda 使用 CPU

示例
----
  # 用视频测试前 10 秒（如 test.mp4）
  python demo_camera.py --source test.mp4

  # 视频只播前 5 秒
  python demo_camera.py --source test.mp4 --seconds 5

  # 使用默认摄像头
  python demo_camera.py

  # 使用 CPU
  python demo_camera.py --source test.mp4 --no_cuda

按 Q 或 Esc 退出预览窗口。
"""
import argparse
import os
import time
import cv2
import torch
import numpy as np
import scipy.special
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from model.model import parsingNet
from data.constant import tusimple_row_anchor

GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56
ROW_ANCHOR = tusimple_row_anchor


class PID:
    """
    标准 PID 控制器。用于根据横向偏差 error 计算平滑的转向修正量 steer。
    steer：机器人转向修正量。正=向右转，负=向左转，绝对值越大转得越猛。
    可接到舵机角度、差速轮左右轮速差等控制接口。
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev = 0.0
        self.sum = 0.0

    def update(self, error, dt):
        dt = max(dt, 1e-6)  # 避免 dt=0 除零
        self.sum += error * dt
        diff = (error - self.prev) / dt
        self.prev = error
        return self.kp * error + self.ki * self.sum + self.kd * diff


def out_j_to_lanes(out_j, col_sample_w, img_w, img_h):
    """把模型输出 out_j (56, 4) 转成 lanes：每条车道是 56 个 (x, y)，无效点用 (0, y)。"""
    lanes = []
    for lane_idx in range(out_j.shape[1]):
        lane = []
        for k in range(out_j.shape[0]):
            py = int(img_h * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
            if out_j[k, lane_idx] > 0:
                px = int(out_j[k, lane_idx] * col_sample_w * img_w / 800) - 1
                lane.append((px, py))
            else:
                lane.append((0, py))
        lanes.append(lane)
    return lanes


def compute_center(lanes, img_width):
    """
    求「当前车道」中心线（本车所在车道的中心），不是整条路的中心。
    思路：画面中心 ≈ 车头方向，选横坐标刚好夹住画面中心的两条车道线，
    作为本车道的左右边界，再取这两条线之间的中线即为当前车道中心线。
    """
    center_x = img_width / 2
    # 每条车道一个平均横坐标（用有效点）
    lane_avg_x = []
    for lane in lanes:
        valid_x = [p[0] for p in lane if p[0] > 0]
        if not valid_x:
            lane_avg_x.append(None)
            continue
        lane_avg_x.append(np.mean(valid_x))
    # 左边界：平均 x 在画面中心左侧且尽量靠右（即离中心最近的那条左线）
    left = None
    left_best = -1
    for i, avg_x in enumerate(lane_avg_x):
        if avg_x is None or avg_x >= center_x:
            continue
        if avg_x > left_best:
            left_best = avg_x
            left = lanes[i]
    # 右边界：平均 x 在画面中心右侧且尽量靠左
    right = None
    right_best = img_width + 1
    for i, avg_x in enumerate(lane_avg_x):
        if avg_x is None or avg_x < center_x:
            continue
        if avg_x < right_best:
            right_best = avg_x
            right = lanes[i]
    if left is None or right is None:
        return None
    center = []
    for l, r in zip(left, right):
        if l[0] > 0 and r[0] > 0:
            cx = (l[0] + r[0]) / 2
            cy = (l[1] + r[1]) / 2
            center.append((int(cx), int(cy)))
    return center


def lateral_error(center, img_width):
    """
    用「最底部」一个中心点（最接近车/机器人）算横向偏差。
    center 顺序：center[0]=画面最下方（靠近车），center[-1]=画面最上方（远处）。
    故取 center[0] 作为最接近机器人的点。
    return: error > 0 → 偏右，error < 0 → 偏左，error = 0 → 居中
    """
    if center is None or len(center) == 0:
        return 0
    cx = center[0][0]  # 最底部一点 = 最接近车
    camera_center = img_width / 2
    error = cx - camera_center
    return error


def draw_instruction(vis, text, img_h, font_size=28, extra_line=None):
    """在画面左下角画红色指令文字，支持中文。extra_line 为第二行（如 steer）。"""
    try:
        # 优先用系统中文字体（Windows 常见）
        for name in ["msyh.ttc", "simhei.ttf", "simsun.ttc"]:
            for d in ["C:/Windows/Fonts", "/usr/share/fonts"]:
                path = os.path.join(d, name)
                if os.path.isfile(path):
                    font = ImageFont.truetype(path, font_size)
                    break
            else:
                continue
            break
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # PIL 在 RGB 上画，再转回 BGR
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil)
    x, y = 20, img_h - 50
    draw.text((x, y), text, fill=(255, 0, 0), font=font)
    if extra_line is not None:
        draw.text((x, img_h - 90), extra_line, fill=(255, 0, 0), font=font)
    vis[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def parse_args():
    parser = argparse.ArgumentParser(description='摄像头/视频实时车道线检测')
    parser.add_argument('--model', type=str, default='weights/tusimple_18.pth', help='模型权重路径')
    parser.add_argument('--source', type=str, default=None, help='视频文件路径，不传则用摄像头')
    parser.add_argument('--seconds', type=float, default=10.0, help='仅对视频有效：只播放前 N 秒（默认 10）')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备号，0 为默认')
    parser.add_argument('--no_cuda', action='store_true', help='使用 CPU')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # 加载模型
    net = parsingNet(
        pretrained=False,
        backbone='18',
        cls_dim=(GRIDING_NUM + 1, CLS_NUM_PER_LANE, 4),
        use_aux=False,
    ).to(device)
    state_dict = torch.load(args.model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        compatible_state_dict[k[7:] if 'module.' in k else k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    col_sample = np.linspace(0, 800 - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    use_video = args.source is not None
    if use_video:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise SystemExit('无法打开视频: ' + args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(fps * args.seconds)
        print(f'视频已打开，只播放前 {args.seconds} 秒（约 {max_frames} 帧），按 Q 或 Esc 退出')
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit('无法打开摄像头，请检查设备号（如 --camera 0）')
        max_frames = None
        print('摄像头已打开，按 Q 或 Esc 退出')
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('Lane Detection (Q/Esc to quit)', cv2.WINDOW_NORMAL)
    frame_count = 0
    pid = PID(0.01, 0.0, 0.002)
    t_prev = time.perf_counter()
    # 工程要点：滤波防抖、限幅防暴走、低频控制
    prev_error = 0.0
    steer = 0.0
    CONTROL_HZ = 10
    control_interval = 1.0 / CONTROL_HZ
    last_control_time = t_prev

    with torch.no_grad():
        while True:
            if use_video and max_frames is not None and frame_count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = img_transform(img_pil).unsqueeze(0).to(device)
            out_t = net(x)

            out_j = out_t[0].cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(GRIDING_NUM) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == GRIDING_NUM] = 0
            out_j = loc

            vis = frame.copy()
            # 画车道绿点
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            px = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                            py = int(img_h * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
                            cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
            # 画面正中央一根蓝色竖线（视觉中心），用于和红线对比偏左/偏右
            cx_cam = int(img_w / 2)
            cv2.line(vis, (cx_cam, 0), (cx_cam, img_h), (255, 0, 0), 2)
            # 转成 lanes 并画中心线（红线），只强调最底部一个红点（最接近机器人）
            lanes = out_j_to_lanes(out_j, col_sample_w, img_w, img_h)
            center = compute_center(lanes, img_w)
            if center and len(center) > 1:
                for i in range(len(center) - 1):
                    cv2.line(vis, center[i], center[i + 1], (0, 0, 255), 3)
                # 最底部一个红点（center[0]=画面最下方，最接近车）：用大圆标出
                bottom_pt = center[0]
                cv2.circle(vis, bottom_pt, 12, (0, 0, 255), -1)
            # 1) 原始横向偏差
            raw_error = lateral_error(center, img_w)
            # 2) 滤波：error = 0.8*prev + 0.2*new，避免检测抖动导致机器人摇摆
            filtered_error = 0.8 * prev_error + 0.2 * raw_error
            prev_error = filtered_error
            t_now = time.perf_counter()
            # 3) 低频控制：视觉可 30FPS，控制 10Hz 即可，不要每帧都控
            if t_now - last_control_time >= control_interval:
                dt_control = t_now - last_control_time
                steer = pid.update(filtered_error, dt_control)
                # 4) 限幅：防止暴走
                steer = max(min(steer, 1.0), -1.0)
                last_control_time = t_now
            t_prev = t_now
            # 显示用滤波后的 error，steer 为 10Hz 控制量
            if center is None or len(center) == 0:
                cmd = "无车道"
            elif filtered_error > 0:
                cmd = "偏右 (%.0f)" % filtered_error
            elif filtered_error < 0:
                cmd = "偏左 (%.0f)" % filtered_error
            else:
                cmd = "居中 (0)"
            draw_instruction(vis, cmd, img_h, extra_line="steer: %.3f (10Hz)" % steer)
            cv2.imshow('Lane Detection (Q/Esc to quit)', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
