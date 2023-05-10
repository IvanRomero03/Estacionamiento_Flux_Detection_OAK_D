import cv2
import depthai as dai
import numpy as np
import blobconverter
# from depthai_sdk import FPSHandler
import os

WS_PATH = os.getcwd()
MODEL_PATH = os.path.join(WS_PATH, "models")
YOLO_BLOB = os.path.join(MODEL_PATH, "yolov4_tiny.blob") 
DETC_BLOB = os.path.join(MODEL_PATH, "license_plate_detection.blob")
OCR_BLOB = os.path.join(MODEL_PATH, "license_plate_recognition.blob")


# Define the pipeline for the device connection
pipeline = dai.Pipeline()

# dai.OpenVINO.VERSION_2020_3
# dai.OpenVINO.VERSION_2020_4
# dai.OpenVINO.VERSION_2021_1
# dai.OpenVINO.VERSION_2021_2
# dai.OpenVINO.VERSION_2021_3
# dai.OpenVINO.VERSION_2021_4
# dai.OpenVINO.VERSION_2022_1
#pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
#supported = dai.OpenVINO.getBlobSupportedVersions(YOLO_BLOB)
# supported = dai.OpenVINO.getVersionName(YOLO_BLOB)
# print(supported)
# pipeline.setOpenVINOVersion(version=dai.OpenVINO.getBlobSupportedVersions())
#pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)
# Camera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(672,384)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Depht node
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
xout_depth = pipeline.createXLinkOut()

## Left camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
## Right camera
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

## Stereo setup
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

## Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# License plate detection
detection_nn = pipeline.createMobileNetDetectionNetwork()
# C:\\Users\\iwell\\.cache\\blobconverter\\vehicle-license-plate-detection-barrier-0106_openvino_2021.4_4shave.blob
detection_nn.setBlobPath(DETC_BLOB)
#detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106"))
detection_nn.setConfidenceThreshold(0.5)
detection_nn.input.setBlocking(False)
xout_detection = pipeline.createXLinkOut()
xout_detection.setStreamName("detection")
detection_nn.out.link(xout_detection.input)

# # Yolo node (Cars detection)
# yoloDet = pipeline.createYoloDetectionNetwork()
# yoloDet.setBlobPath(YOLO_BLOB)
# yoloDet.setConfidenceThreshold(0.5)
# yoloDet.setNumClasses(80)
# yoloDet.setCoordinateSize(4)
# yoloDet.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
# yoloDet.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
# yoloDet.setIouThreshold(0.5)

# ## Linking
# cam_rgb.preview.link(yoloDet.input)
# yoloDet.out.link(xout_detection.input)
# xout_yolo = pipeline.createXLinkOut()
# xout_yolo.setStreamName("yolo")
# yoloDet.out.link(xout_yolo.input)

# OCR node
text_recognition_nn = pipeline.createNeuralNetwork()
# C:\\Users\\iwell\\.cache\\blobconverter\\license-plate-recognition-barrier-0007_openvino_2021.4_4shave.blob
text_recognition_nn.setBlobPath(OCR_BLOB)
#text_recognition_nn.setBlobPath(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007"))
text_recognition_nn.input.setBlocking(False)

## Linking
xout_ocr = pipeline.createXLinkOut()
xout_ocr.setStreamName("ocr")
text_recognition_nn.out.link(xout_ocr.input)

# fps = FPSHandler()

frame_rgb = None
frame_depth = None
detections = []
yolo_detections = []
ocr_detections = []

# Start pipeline runtime
print("Starting pipeline")
with dai.Device(pipeline) as device:
    print("Pipeline started")
    queue_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print("Got rgb queue")
    queue_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    print("Got depth queue")
    queue_detection = device.getOutputQueue(name="detection", maxSize=1, blocking=False)
    # queue_yolo = device.getOutputQueue(name="yolo", maxSize=4, blocking=False)
    print("Got detection queue")
    queue_ocr = device.getOutputQueue(name="ocr", maxSize=1, blocking=False)
    print("Got ocr queue")
    while True:
        print("Getting data from queue")
        in_rgb = queue_rgb.get()
        print("Got rgb frame")
        in_depth = queue_depth.get()
        print("Got depth frame")
        in_detection = queue_detection.get()
        print("Got detection")
        # in_yolo = queue_yolo.get()
        in_ocr = queue_ocr.get()

        print("Got data from queue")
        if in_rgb is not None:
            print("Got rgb frame")
            frame_rgb = in_rgb.getCvFrame()
            print("Got rgb frame")
        if in_depth is not None:
            print("Got depth frame")
            frame_depth = in_depth.getFrame()
            print("Got depth frame")

        if in_detection is not None:
            print("Got detection")
            detections = in_detection.detections
            print("Got detection")

        # if in_yolo is not None:
        #     print("Got yolo detection")
        #     yolo_detections = in_yolo.detections
        #     print("Got yolo detection")

        if in_ocr is not None:
            print("Got ocr detection")
            ocr_detections = in_ocr.detections
            print("Got ocr detection")

        # Display the frame
        if frame_rgb is not None:
            print("Displaying frame")
            # Display the frame
            cv2.imshow("rgb", frame_rgb)

            # Display the depth
            if frame_depth is not None:
                frame_depth = cv2.normalize(frame_depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                frame_depth = cv2.equalizeHist(frame_depth)
                frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_HOT)
                cv2.imshow("depth", frame_depth)

            # Display the detections
            if len(detections) > 0:
                for detection in detections:
                    bbox = detection.bbox
                    cv2.rectangle(frame_rgb, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (255, 0, 0), 2)
                    cv2.putText(frame_rgb, str(detection.label), (bbox.xmin + 10, bbox.ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            # Display the yolo detections
            if len(yolo_detections) > 0:
                for detection in yolo_detections:
                    bbox = detection.bbox
                    cv2.rectangle(frame_rgb, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, str(detection.label), (bbox.xmin + 10, bbox.ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            # Display the ocr detections
            if len(ocr_detections) > 0:
                for detection in ocr_detections:
                    bbox = detection.bbox
                    cv2.rectangle(frame_rgb, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 0, 255), 2)
                    cv2.putText(frame_rgb, str(detection.label), (bbox.xmin + 10, bbox.ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            # Show the frame
            cv2.imshow("rgb", frame_rgb)

        # FPS
        # fps.update()
        # fps.printFPS()
        if cv2.waitKey(1) == ord('q'):
            break