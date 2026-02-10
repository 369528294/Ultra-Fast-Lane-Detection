"""
对单个视频文件进行车道线检测，并输出画好车道线的视频。

使用说明
--------
必选参数：
  --model   权重文件路径（如 Tusimple 预训练模型）
  --source  输入视频路径

可选参数：
  --output  输出视频路径。不指定时，默认在输入文件名后加 _lanes.mp4
  --no_cuda 加上此参数则强制使用 CPU（默认：有 GPU 用 GPU，没有则用 CPU）

示例
----
  # 对 test.mp4 检测，输出 test_lanes.mp4
  python demo_video.py --model weights/tusimple_18.pth --source test.mp4

  # 指定输出文件名
  python demo_video.py --model weights/tusimple_18.pth --source test.mp4 --output result.mp4

  # 无 NVIDIA 显卡或想用 CPU 时
  python demo_video.py --model weights/tusimple_18.pth --source test.mp4 --no_cuda
"""
import argparse
import os
import cv2
import torch
import numpy as np
import scipy.special
import tqdm
from PIL import Image
from torchvision import transforms

from model.model import parsingNet
from data.constant import tusimple_row_anchor

# ---------- Tusimple 模型相关常数（与训练/官方 demo 一致）----------
GRIDING_NUM = 100      # 横向网格数，用于预测车道点横坐标
CLS_NUM_PER_LANE = 56  # 每条车道在纵向上的分类数（行锚数量）
ROW_ANCHOR = tusimple_row_anchor  # 行锚纵坐标（在 288 高度下的相对位置）


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='对视频做车道线检测')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径，如 weights/tusimple_18.pth')
    parser.add_argument('--source', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径，默认：输入文件名_lanes.mp4')
    parser.add_argument('--no_cuda', action='store_true', help='强制使用 CPU')
    return parser.parse_args()


def main():
    args = parse_args()
    # 未指定输出路径时，在输入文件名后加 _lanes.mp4
    args.output = args.output or os.path.splitext(args.source)[0] + '_lanes.mp4'

    # 有 CUDA 且未指定 --no_cuda 时用 GPU，否则用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # ---------- 加载模型 ----------
    net = parsingNet(
        pretrained=False,
        backbone='18',
        cls_dim=(GRIDING_NUM + 1, CLS_NUM_PER_LANE, 4),
        use_aux=False,
    ).to(device)
    state_dict = torch.load(args.model, map_location='cpu')['model']
    # 兼容多卡训练保存的权重（键名带 "module." 前缀）
    compatible_state_dict = {}
    for k, v in state_dict.items():
        compatible_state_dict[k[7:] if 'module.' in k else k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # 与训练时一致的图像预处理：缩放到 288x800 + 归一化
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 横向采样点，用于把模型输出的网格索引转成像素横坐标
    col_sample = np.linspace(0, 800 - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    # ---------- 打开输入/输出视频 ----------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit('无法打开视频: ' + args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (img_w, img_h))

    # ---------- 逐帧检测并画车道点 ----------
    with torch.no_grad():
        for _ in tqdm.tqdm(range(total), desc='处理中'):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB，转 PIL 后做与训练一致的变换
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = img_transform(img_pil).unsqueeze(0).to(device)
            out_t = net(x)

            # 将模型输出从“网格分类”解码为“横向位置”（与 demo.py 一致）
            out_j = out_t[0].cpu().numpy()
            out_j = out_j[:, ::-1, :]  # 左右翻转
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(GRIDING_NUM) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == GRIDING_NUM] = 0  # 背景类不画
            out_j = loc

            # 在原图上画车道点：每列有效点超过 2 个才画，绿点
            vis = frame.copy()
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            px = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                            py = int(img_h * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
                            cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
            out.write(vis)

    cap.release()
    out.release()
    print('已保存:', args.output)


if __name__ == '__main__':
    main()
