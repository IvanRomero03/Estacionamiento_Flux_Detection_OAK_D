import time
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np

# Paths to model files
WS_PATH = Path(__file__).resolve().parent
MODELS_PATH = WS_PATH.joinpath("models")
YOLO_BLOB = MODELS_PATH.joinpath("yolov4_tiny.blob")
DETC_BLOB = MODELS_PATH.joinpath("license_plate_detection.blob")
OCR_BLOB = MODELS_PATH.joinpath("license_plate_recognition.blob")

ITEMS_REC = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>", "<Gansu>",
         "<Guangdong>", "<Guangxi>", "<Guizhou>", "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>", "<HongKong>",
         "<Hubei>", "<Hunan>", "<InnerMongolia>", "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>", "<Macau>",
         "<Ningxia>", "<Qinghai>", "<Shaanxi>", "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>", "<Tianjin>",
         "<Tibet>", "<Xinjiang>", "<Yunnan>", "<Zhejiang>", "<police>", "A", "B", "C", "D", "E", "F", "G", "H", "I",
         "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Creating Nodes
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setPreviewSize(672, 384)
rgb_cam.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setInterleaved(False)
rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
## Linking
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam_out")
rgb_cam.preview.link(xout_rgb.input)

# Car detection model
car_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
car_detection_nn.setConfidenceThreshold(0.5)
car_detection_nn.setBlobPath(blobconverter.from_zoo(
    name="vehicle-detection-adas-0002", shaves=6))
car_detection_nn.input.setQueueSize(1)
car_detection_nn.input.setBlocking(False)
xout_car_det = pipeline.create(dai.node.XLinkOut)
xout_car_det.setStreamName("car_detection_nn")
car_detection_nn.out.link(xout_car_det.input)
rgb_cam.preview.link(car_detection_nn.input)

# License plate recognition model
license_plate_recognition_nn = pipeline.create(dai.node.NeuralNetwork)
license_plate_recognition_nn.setBlobPath(blobconverter.from_zoo(
    name="license-plate-recognition-barrier-0007", shaves=6))
license_plate_recognition_nn.input.setBlocking(False)
license_plate_recognition_nn.input.setQueueSize(1)
xout_license_plate_recognition = pipeline.create(dai.node.XLinkOut)
xout_license_plate_recognition.setStreamName("license_plate_recognition_nn")
license_plate_recognition_nn.out.link(xout_license_plate_recognition.input)

# Linking
car_detection_nn.passthrough.link(license_plate_recognition_nn.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="cam_out", maxSize=1, blocking=True)
    q_car = device.getOutputQueue(name="car_detection_nn", maxSize=1, blocking=False)
    q_license_plate = device.getOutputQueue(name="license_plate_recognition_nn", maxSize=1, blocking=False)

    frame = None
    detections = []
    license_plate = None
    while True:
        in_rgb = q_rgb.get()
        in_car = q_car.get()
        in_license_plate = q_license_plate.get()

        # Show the frame
        frame = in_rgb.getCvFrame()
        cv2.imshow("rgb", frame)

        # Get the car detections
        detections = in_car.detections
        # Iterate through all detections
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * frame.shape[1])
            x2 = int(detection.xmax * frame.shape[1])
            y1 = int(detection.ymin * frame.shape[0])
            y2 = int(detection.ymax * frame.shape[0])
            # Show the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Get the license plates
        licences_recs = []
        try:
            rec_data = in_license_plate.getFirstLayerInt32()
            print("[LOG] rec_data: ---------------------------------------------------------")
            print(rec_data)
            print("[LOG] end ---------------------------------------------------------")
            decoded_text = ""

            for idx in rec_data:
                if idx == -1:
                    break
                decoded_text += ITEMS_REC[int(idx)]
            print("[LOG] decoded_text: " + decoded_text)
            licences_recs = [(cv2.resize(frame, (200, 64)),
                            decoded_text)] + licences_recs[:9]  
            
            # Iterate through all license plates
            for license_plate in licences_recs:
                # Show the license plate
                cv2.imshow("license_plate", license_plate[0])
                # Show the license plate text
                print(license_plate[1])
        except RuntimeError:
             # Show the frame
            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
            continue
            

        # Show the frame
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

