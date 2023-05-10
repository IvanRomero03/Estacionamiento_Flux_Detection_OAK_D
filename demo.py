import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()
yoloDet = pipeline.createYoloDetectionNetwork()
yoloDet.setBlobPath("C:\\Users\\iwell\\.cache\\blobconverter\\yolo-v3-tiny-tf_openvino_2021.4_6shave.blob")

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

# Stereo node
camLeft = pipeline.createMonoCamera()
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

camRight = pipeline.createMonoCamera()
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(200)

# Camera node
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Yolo node
yoloDet.setConfidenceThreshold(0.5)
yoloDet.setNumClasses(80)
yoloDet.setCoordinateSize(4)
yoloDet.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
yoloDet.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
yoloDet.setIouThreshold(0.5)


# Linking
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")

camRgb.preview.link(yoloDet.input)
yoloDet.passthrough.link(xoutRgb.input)
yoloDet.out.link(nnOut.input)

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

with dai.Device(pipeline) as device:
    qYolo = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inYolo = qYolo.get()
        detections = np.array(inYolo.detections)
        cars = []
        for detection in detections:
            if detection.label == 2:
                cars.append(detection)
                print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                print(detection.confidence)
                print(detection.label)

        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        inDepth = qDepth.get()
        depthFrame = inDepth.getFrame()
        # Heatmap
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        for car in cars:
            # get car attributes
            print(dir(car))
            print(car.xmin, car.ymin, car.xmax, car.ymax)
            print(car.confidence)
            print(car.label)
            bbox = frameNorm(frame, [car.xmin, car.ymin, car.xmax, car.ymax])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{car.label}", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
            cv2.putText(frame, f"{car.confidence:.2f}", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

            # depth bbox
            depthBbox = frameNorm(depthFrame, [car.xmin, car.ymin, car.xmax, car.ymax])
            cv2.rectangle(depthFrameColor, (depthBbox[0], depthBbox[1]), (depthBbox[2], depthBbox[3]), (0, 255, 0), 2)
            cv2.putText(depthFrameColor, f"{car.label}", (depthBbox[0] + 10, depthBbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
            cv2.putText(depthFrameColor, f"{car.confidence:.2f}", (depthBbox[0] + 10, depthBbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
        
        #images = np.hstack((frame, depthFrameColor))
        #cv2.imshow("depth", images)
        


        cv2.imshow("rgb", frame)
        cv2.imshow("depth", depthFrameColor)
        if cv2.waitKey(1) == ord('q'):
            break

