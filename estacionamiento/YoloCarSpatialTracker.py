import cv2
import depthai as dai
import numpy as np
import os
import blobconverter

global_counter = 0

MODEL_PATH = "../models/yolov8n_openvino_2022.1_6shave.blob"
# MODEL_PATH = "../models/yolov8m_openvino_2022.1_6shave.blob"
# MODEL_PATH = blobconverter.from_zoo("yolo-v3-tiny-tf", shaves=6)
# Define the pipeline for nodes
pipeline = dai.Pipeline()

# Set nodes
rgb_node = pipeline.createColorCamera()
spatial_yolo_node = pipeline.createYoloSpatialDetectionNetwork()
mono_left_node = pipeline.createMonoCamera()
mono_right_node = pipeline.createMonoCamera()
stereo_node = pipeline.createStereoDepth()
car_tracker_node = pipeline.createObjectTracker()

# Set nodes output stream
xout_rgb = pipeline.createXLinkOut()
xout_car_tracker = pipeline.createXLinkOut()

xout_rgb.setStreamName("rgb")
xout_car_tracker.setStreamName("car_tracker")

# Define nodes properties
# Cam
rgb_node.setPreviewSize(416, 416)
rgb_node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
rgb_node.setInterleaved(False)
rgb_node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Stereo
mono_left_node.setResolution(
    dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left_node.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right_node.setResolution(
    dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right_node.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo_node.setDefaultProfilePreset(
    dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_node.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo_node.setOutputSize(rgb_node.getPreviewWidth(),
                          rgb_node.getPreviewHeight())

# Spatial detection
spatial_yolo_node.setBlobPath(MODEL_PATH)
spatial_yolo_node.setConfidenceThreshold(0.85)
spatial_yolo_node.input.setBlocking(False)
spatial_yolo_node.setBoundingBoxScaleFactor(0.5)
spatial_yolo_node.setDepthLowerThreshold(100)
spatial_yolo_node.setDepthUpperThreshold(5000)
spatial_yolo_node.setNumClasses(80)
spatial_yolo_node.setCoordinateSize(4)
spatial_yolo_node.setAnchors(
    [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
spatial_yolo_node.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
spatial_yolo_node.setIouThreshold(0.5)

# Car tracker
car_tracker_node.setDetectionLabelsToTrack([2])
car_tracker_node.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# car_tracker_node.setTrackerIdAssigmentPolicy(
#     dai.TrackerIdAssigmentPolicy.UNIQUE_ID)
# car_tracker_node.setTrackerIdAssignmentPolicy(
#     dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

# Link nodes
mono_left_node.out.link(stereo_node.left)
mono_right_node.out.link(stereo_node.right)
rgb_node.preview.link(spatial_yolo_node.input)
car_tracker_node.passthroughTrackerFrame.link(xout_rgb.input)
car_tracker_node.out.link(xout_car_tracker.input)

rgb_node.setPreviewKeepAspectRatio(False)
rgb_node.video.link(car_tracker_node.inputTrackerFrame)
car_tracker_node.inputTrackerFrame.setBlocking(False)
car_tracker_node.inputTrackerFrame.setQueueSize(2)

# car_tracker_node.passthrough.link(car_tracker_node.inputTrackerFrame)

spatial_yolo_node.passthrough.link(car_tracker_node.inputDetectionFrame)
spatial_yolo_node.out.link(car_tracker_node.inputDetections)
stereo_node.depth.link(spatial_yolo_node.inputDepth)


# Logic
# global_counter = 0

tracker_depth_origin = {
    -1: 10,
}

tracker_id_frames_count = {
    -1: 0,
}

DEPTH_FRONTIER = 500
FRAMES_THRESHOLD = 20

registered_cars = []

# Use device
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    car_tracker_queue = device.getOutputQueue(
        name="car_tracker", maxSize=4, blocking=False)

    while True:
        rgb_image = rgb_queue.get()
        cars_trackings = car_tracker_queue.get()

        frame = rgb_image.getCvFrame()
        # print(dir(cars_trackings))
        tracked_cars = cars_trackings.tracklets

        for car in tracked_cars:
            # print(car)
            bbox = car.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(bbox.topLeft().x)
            y1 = int(bbox.topLeft().y)
            x2 = int(bbox.bottomRight().x)
            y2 = int(bbox.bottomRight().y)
            z = car.spatialCoordinates.z
            if (car.status.name != "LOST"):
                if (car.id not in tracker_depth_origin):
                    print("New car detected at: " + str(z) +
                          "mm" + " with id: " + str(car.id))
                    tracker_depth_origin[car.id] = z
                    tracker_id_frames_count[car.id] = 0
                else:
                    if (car.status.name != "LOST"):
                        tracker_id_frames_count[car.id] += 1
                        if (tracker_id_frames_count[car.id] >= FRAMES_THRESHOLD) and (z - tracker_depth_origin[car.id] >= DEPTH_FRONTIER) and (car.id not in registered_cars):

                            registered_cars.append(car.id)
                            global_counter += 1
                            # r.set("counter", global_counter)

                # Normal Bbox
                cv2.putText(frame, "car", (x1 + 10, 71 + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, car.status.name, (x1 + 10, y1 + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                # Id Label
                cv2.putText(frame, "ID: " + str(car.id), (x1 + 10, y1 + 35),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                # Spatial cords
                cv2.putText(frame, "X: " + str(car.spatialCoordinates.x),
                            (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "Y: " + str(car.spatialCoordinates.y),
                            (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "Z: " + str(car.spatialCoordinates.z),
                            (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # Counter (top right)
        cv2.putText(frame, "Counter: " + str(global_counter), (10, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, 255, thickness=2)

        # cv2.imshow("Demo", frame)
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("demo", 1280, 720)
        cv2.imshow("demo", frame)

        if cv2.waitKey(1) == ord('q'):
            break
