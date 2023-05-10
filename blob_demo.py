import cv2
import depthai as dai
import numpy as np

# Define the threshold value for detecting white pixels
threshold = 200

# Create a pipeline that connects to the OAK-D-Lite camera
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setFps(30)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("rgb")
cam.preview.link(cam_xout.input)

# Create a blob detector object
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 10
detector = cv2.SimpleBlobDetector_create(params)

# Start the pipeline and get the output stream
with dai.Device(pipeline) as device:
    # is this running on the device or on the host?
    if device.getUsbSpeed() == dai.UsbSpeed.SUPER:
        print("Running on the device")
    else:
        print("Running on the host")
    q = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    while True:
        # Get the next frame from the camera
        in_rgb = q.get()
        frame = in_rgb.getCvFrame()

        # Convert the frame to grayscale and apply thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Detect blobs in the thresholded image
        keypoints = detector.detect(thresh)

        # Draw the detected blobs on the original color frame
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show the result
        cv2.imshow("White Blobs Detection", im_with_keypoints)

        # Wait for key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
