import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
# 先不加入spatial

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName("left")  # 常规视觉包
xoutRight.setStreamName("right")  # 常规视觉包
xoutDepth.setStreamName("stereo")  # 深度信息包

# 物理层config
monoLeft.setCamera("left")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setCamera("right")
monoRight.setCamera(dai.MonoCameraProperties.SensorResolution.THE_720_P)

stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)  # 从不同方向计算两次图像差异，减少错误
# https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth/#StereoDepth-Currently%20configurable%20blocks
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

# link the device
monoLeft.out.link(xoutLeft.input)  # 拓扑链接,out连input
monoRight.out.link(xoutRight.input)
monoLeft.out.link(stereo.left) # 将monocam信息同时输出到stereo部分
monoRight.out.link(stereo.right)


with dai.Device(pipeline) as device:
    qLeft = device.getOutputQueue(name="left", blocking=False)
