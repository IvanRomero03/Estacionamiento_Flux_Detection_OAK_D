import depthai
import cv2
import numpy as np

# Create a pipeline
pipeline = depthai.Pipeline()

# Define a node for the color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(416, 416)
cam.setInterleaved(False)
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
cam.setFps(30)
cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("rgb")
cam.preview.link(cam_xout.input)

# Define a node for the YOLOv4 model
detection_nn = pipeline.createYoloDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors([
    10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
    30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
    116.0, 90.0, 156.0, 198.0, 373.0, 326.0
])
detection_nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
#detection_nn.setOutSize(1)
detection_nn.setBlobPath("C:\\Users\\iwell\\.cache\\blobconverter\\yolo-v3-tiny-tf_openvino_2021.4_6shave.blob")
cam.preview.link(detection_nn.input)

# Define a node for encoding the detection results as JSON
detection_encoder = pipeline.createXLinkOut()
detection_encoder.setStreamName("detections")
detection_nn.out.link(detection_encoder.input)

# Start the pipeline
with depthai.Device(pipeline) as device:
    # Get the output queues for the nodes
    cam_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    detections_queue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)

    while True:
        # Get the next frame from the camera
        in_rgb = cam_queue.get()
        frame = in_rgb.getCvFrame()

        # Get the next set of detections
        in_detections = detections_queue.get()
        
        #detections = np.array(in_detections.getFirstLayerFp16())
        #detections = np.array(in_detections.getOutputs())
        print(in_detections.getData())
        detections = np.array(in_detections.getData())
        #detections = np.array(in_detections.getDetectionResults())
        detections = detections.reshape((detections.size // 6, 6))

        # Filter the detections to only include cars
        cars = []
        for detection in detections:
            if detection[1] == 2:  # 2 is the class ID for cars
                x1, y1, x2, y2 = detection[2:6]
                cars.append((x1, y1, x2, y2))

        # Draw bounding boxes around the detected cars
        for car in cars:
            x1, y1, x2, y2 = car
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Show the result
        cv2.imshow("Car Detection", frame)

        # Wait for key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
