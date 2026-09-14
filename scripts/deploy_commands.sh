#!/usr/bin/env bash
# Jetson Xavier NX 上 YOLOv5 + TensorRT 部署命令（实验记录整理，按需修改路径与设备 IP）
set -e

DEVICE_IP="${1:-192.168.0.100}"

echo "==> 1. SSH 登录设备（用户名/密码：nvidia/nvidia）"
ssh "nvidia@${DEVICE_IP}" bash -lc '
set -e
echo "==> 2. 克隆已配置好的 conda 环境"
conda create -n 02_env --clone test -y
source activate 02_env

echo "==> 3. 准备工作目录"
mkdir -p ~/stu_workspace/02
cp -r ~/Monday_Morning/yolov5 ~/stu_workspace/02/
cp -r ~/Monday_Morning/tensorrtx ~/stu_workspace/02/
cd ~/stu_workspace/02/yolov5

echo "==> 4. PyTorch 基线推理"
python detect.py

echo "==> 5. 生成 wts"
cp ../tensorrtx/yolov5/gen_wts.py ./
python gen_wts.py -w yolov5s.pt -o yolov5s.wts

echo "==> 6. 编译 tensorrtx 并生成 engine"
cd ../tensorrtx/yolov5/
mkdir -p build && cd build
cp ../../../yolov5/yolov5s.wts ./
sudo apt update && sudo apt install -y cmake
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s

echo "==> 7. 用 engine 推理"
cp -r ../../../yolov5/data/images/ ../images
sudo ./yolov5 -d yolov5s.engine ../images
'
