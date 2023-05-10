import os
import blobconverter
import cv2
import depthai as dai

pipeline = dai.Pipeline()

### MODELS ###
# Car detection model:
# yolo-v4-tiny-tf
# License plate Detection model:
# vehicle-license-plate-detection-barrier-0123
# OCR model:
# text-recognition-resnet-fc

# RGB camera
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setPreviewSize(416, 416)
rgb_cam.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setInterleaved(False)
rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
# Linking
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam_out")
rgb_cam.preview.link(xout_rgb.input)

# Car detection node (Yolov4 tiny)
# Car indx = 2
yolo_nn = pipeline.createYoloDetectionNetwork()
yolo_nn.setConfidenceThreshold(0.8)
yolo_nn.setNumClasses(80)
yolo_nn.setCoordinateSize(4)
yolo_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
yolo_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
yolo_nn.setIouThreshold(0.5)
yolo_nn.setBlobPath(blobconverter.from_zoo(
    name="yolo-v3-tiny-tf", shaves=6))

xout_yolo = pipeline.createXLinkOut()
xout_yolo.setStreamName("yolo_nn")
rgb_cam.preview.link(yolo_nn.input)
yolo_nn.out.link(xout_yolo.input)


# License detection node
det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
det_nn.setConfidenceThreshold(0.2)
det_nn.setBlobPath(blobconverter.from_zoo(
    name="vehicle-license-plate-detection-barrier-0106", shaves=6))
det_nn.input.setQueueSize(1)
det_nn.input.setBlocking(False)
xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("det_nn")
rgb_cam.preview.link(det_nn.input)
det_nn.out.link(xout_det.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
    q_yolo = device.getOutputQueue(name="yolo_nn", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det_nn", maxSize=4, blocking=False)

    frame = None
    while True:
        detections = []
        plates = []

        in_rgb = q_rgb.get()
        in_yolo = q_yolo.get()
        in_plates = q_det.get()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            if frame is not None:
                cv2.imshow("rgb", frame)

        if in_yolo is not None:
            detections = in_yolo.detections

        if in_plates is not None:
            print("[DEBUG] Plates: ", in_plates.detections)
            plates = in_plates.detections

        if frame is not None:
            for detection in detections:
                if detection.label == 2:
                    x1 = int(detection.xmin * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    x2 = int(detection.xmax * frame.shape[1])
                    y2 = int(detection.ymax * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for plate in plates:
                print("[DEBUG] Plate: ", plate.label)
                # print(plate.__dict__)
                x1 = int(plate.xmin * frame.shape[1])
                y1 = int(plate.ymin * frame.shape[0])
                x2 = int(plate.xmax * frame.shape[1])
                y2 = int(plate.ymax * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
