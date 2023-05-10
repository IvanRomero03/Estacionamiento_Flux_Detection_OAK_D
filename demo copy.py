import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()
yoloDet = pipeline.createYoloSpatialDetectionNetwork()
yoloDet.setBlobPath("C:\\Users\\iwell\\.cache\\blobconverter\\yolo-v3-tiny-tf_openvino_2021.4_6shave.blob")

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Spatial detection specific parameters
yoloDet.setConfidenceThreshold(0.5)
yoloDet.input.setBlocking(False)
yoloDet.setBoundingBoxScaleFactor(0.5)
yoloDet.setDepthLowerThreshold(100) # Min 10 centimeters
yoloDet.setDepthUpperThreshold(5000) # Max 5 meters

# Yolo specific parameters
yoloDet.setNumClasses(80)
yoloDet.setCoordinateSize(4)
yoloDet.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
yoloDet.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
yoloDet.setIouThreshold(0.5)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")


camRgb.preview.link(yoloDet.input)
yoloDet.passthrough.link(xoutRgb.input)
yoloDet.out.link(nnOut.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


print("Starting pipeline...")
with dai.Device(pipeline) as device:
    print("Pipeline started.")
    qYolo = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("Starting inference...")
    while True:
        print("aqui1")
        inYolo = qYolo.get()
        print("aqui3")
        detections = np.array(inYolo.detections)
        print("aqui2")
        cars = []
        for detection in detections:
            if detection.label == 2:
                cars.append(detection)
                print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                print(detection.confidence)
                print(detection.label)

        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        for car in cars:
            # get car attributes
            print(dir(car))
            print(car.xmin, car.ymin, car.xmax, car.ymax)
            print(car.confidence)
            print(car.label)
            bbox = frameNorm(frame, [car.xmin, car.ymin, car.xmax, car.ymax])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("rgb", frame)
        cv2.waitKey(1)
        print("aqui")

