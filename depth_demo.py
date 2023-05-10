import depthai as dai
import cv2
import numpy as np

# Create a Script that shows the depth map
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

camLeft = pipeline.createMonoCamera()
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

camRight = pipeline.createMonoCamera()
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(200)

# Create outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")

# Linking
camRgb.preview.link(xoutRgb.input)
camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.get()

        frame = inRgb.getCvFrame()
        depth = inDepth.getFrame()
        # heatmap
        depth = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth = cv2.equalizeHist(depth)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)

        cv2.imshow("rgb", frame)
        # as heatmap
        cv2.imshow("depth", depth)
        if cv2.waitKey(1) == ord('q'):
            break