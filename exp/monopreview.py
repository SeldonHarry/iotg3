#!/usr/bin/env python3
# 最基本的单目相机例程，理解pipeline、node、message的概念
# 显示的是黑白图片
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName("left")
xoutRight.setStreamName(
    "right"
)  # 不同的stream将在DDS的包上作为唯一区别元素，因此必须不同

# Properties
monoLeft.setCamera("left")  # 设置配置的相机名，是device相机的唯一名称
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setCamera("right")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()

        if inLeft is not None:
            cv2.imshow("left", inLeft.getCvFrame())

        if inRight is not None:
            cv2.imshow("right", inRight.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            break
