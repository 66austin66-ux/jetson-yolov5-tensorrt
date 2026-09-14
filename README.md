# Jetson Xavier NX 上 YOLOv5 的 TensorRT 部署与推理加速

在嵌入式 AI 边缘平台（飞云智盒 X509，基于 NVIDIA Jetson Xavier NX）上部署 YOLOv5，
并用 TensorRT 完成推理加速，对比加速前后的检测速度与精度变化。

**项目信息**

- 课程：数字信息技术专业实验 · 嵌入式 AI 实验五「基于嵌入式 AI 平台的目标检测」
- 时间：2026 年 6 月 1 日
- 形式：小组实验（第 2 组，4 人）
- 硬件：飞云智盒 X509（NVIDIA Jetson Xavier NX，算力约 21 TOPS，实验报告原文写 21 TFLOPS）
- 工具：MobaXterm（SSH 远程）、conda 虚拟环境、YOLOv5、TensorRT、tensorrtx

## 一、实验目标

1. 了解深度神经网络在嵌入式/边缘设备上部署的难点与常见解决方案
2. 理解 TensorRT 的推理加速原理（层融合、显存复用、低精度推理等）
3. 掌握在嵌入式 AI 平台上部署 YOLOv5 并使用 TensorRT 加速的方法

## 二、加速原理（简要）

TensorRT 对已训练模型（ONNX 等）做推理优化，主要手段包括：

- 计算图优化：横向层融合（Conv + Conv）、纵向层融合（Conv + Add + ReLU）
- 节点消除与变换（Pad、Slice、Concat 等无用量/低效节点处理）
- 多精度支持：FP32 / FP16 / INT8 / TF32
- 显存复用：减少显存分配与访存开销
- 为具体硬件选择更优的 kernel 实现

## 三、复现步骤

### 1. 连接设备

设备固定 IP（`192.168.0.x`，以机箱贴纸为准），用 MobaXterm 通过 SSH 登录：

```bash
ssh nvidia@192.168.0.x   # 用户名/密码：nvidia / nvidia
```

### 2. 准备运行环境

Jetson 为 arm 架构，直接配置环境耗时且易出错，因此从已配置好的环境克隆：

```bash
conda create -n 02_env --clone test
conda activate 02_env
mkdir -p stu_workspace/02
cp -r Monday_Morning/yolov5 Monday_Morning/tensorrtx stu_workspace/02/
```

### 3. YOLOv5 原始模型推理（基线）

```bash
cd stu_workspace/02/yolov5
python detect.py          # 使用 COCO 预训练权重 + 官方示例图
```

### 4. PyTorch 权重转 wts（供 tensorrtx 使用）

```bash
cp ../tensorrtx/yolov5/gen_wts.py ./
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
```

### 5. 编译 tensorrtx 并生成 TensorRT engine

```bash
cd ../tensorrtx/yolov5/
mkdir build && cd build
cp ../../../yolov5/yolov5s.wts ./
sudo apt update && sudo apt install cmake -y
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s   # s 表示 yolov5s
```

### 6. 用 TensorRT engine 推理

```bash
cp -r ../../../yolov5/data/images/ ../images
sudo ./yolov5 -d yolov5s.engine ../images
```

## 四、实验结果

| 模型 | 推理精度 | 平均检测速度 |
| --- | --- | --- |
| 原始 YOLOv5s（PyTorch） | FP32 | 约 28 FPS |
| TensorRT 加速（tensorrtx 编译后） | FP32 | 约 42 FPS |

- **速度**：约提升 1.5 倍（折合约 36 ms → 24 ms 单帧）
- **精度**：采用 FP32 未量化，检测精度与原始模型基本一致，未观察到明显精度损失
- 分析：提速主要来自层融合与显存复用（减少 kernel 启动次数与访存开销）以及针对硬件的 kernel 选择；
  FP16/INT8 可进一步提升速度，但需要校准数据，且可能引入轻微精度下降

## 五、仓库说明

本仓库是可复现的**部署与加速记录**，不包含：

- 预训练权重与转换产物（`*.pt`、`*.wts`、`*.engine`，体积大且与设备绑定）
- 数据集与示例图片（使用 YOLOv5 官方示例图，未自采数据、未自行训练模型）
- 智盒上的原始工程文件（实验在设备上完成，未保留完整工程）

注意事项：

- `*.engine` 与生成它的设备/TensorRT 版本绑定，换设备需重新生成
- tensorrtx 分支需与 YOLOv5 版本、TensorRT 版本匹配，否则编译或精度校验会失败
- 转 engine 时最后的参数 `s` 表示按 yolov5s 结构构建，若改用 m/l 需同步修改

## 六、可继续做的方向

- 尝试 FP16 / INT8（需准备校准集）并对比精度与速度
- 使用自采数据集做迁移训练后部署，评估端侧效果
- 接入 RTSP 摄像头做实时的视频流推理

