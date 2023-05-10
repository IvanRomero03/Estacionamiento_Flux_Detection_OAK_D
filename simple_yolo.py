import depthai as dai
from cv2 import imshow, waitKey, destroyAllWindows, rectangle
from numpy import array, reshape

# Pipeline
pipeline = dai.Pipeline()

# Yolo detection pipeline
yoloDet: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
yoloDet.setConfidenceThreshold(0.5)
# Only class for car
yoloDet.setNumClasses(1)
yoloDet.setCoordinateSize(4)
yoloDet.setAnchors(array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
yoloDet.setAnchorMasks({"side26": array([1, 2, 3]), "side13": array([3, 4, 5])})
yoloDet.setIouThreshold(0.5)
yoloDet.setBlobPath("C:\\Users\\iwell\\.cache\\blobconverter\\yolo-v3-tiny-tf_openvino_2021.4_6shave.blob")

# Add a Script node that prints a count
newNode = pipeline.create(dai.node.Script)
newNode.setScript("""
import time
import marshal

# define the state of previous detections
prevDetections = []

node.warn("Starting the script")

def sendSerial(s: str) -> None:
    msg = marshal.dumps(s)
    b = Buffer(len(msg))
    b.setData(msg)
    node.io['out'].send(b)
    node.warn("Sent: " + s)

count = 0

while True:
    time.sleep(1)
    sendSerial("Hello " + str(count))
""")

xout = pipeline.createXLinkOut()
xout.setStreamName("out_xout")
newNode.outputs['out'].link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Get the output queues
    qYolo = device.getOutputQueue(name="yolo", maxSize=4, blocking=False)
    qScript = device.getOutputQueue(name="out", maxSize=4, blocking=False)

    while True:
        frame = qScript.get()
        # Get the next frame from the camera
        inYolo = qYolo.get()
        detections = reshape(inYolo.detections, (inYolo.detections.size // 7, 7))
        # Filter the detections to only include cars
        cars = []
        for detection in detections:
            if detection[2] == 0:
                x1, y1, x2, y2 = detection[3:7]
                cars.append((x1, y1, x2, y2))
            
        # Draw bounding boxes around the detected cars
        for car in cars:
            x1, y1, x2, y2 = car
            rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Show the frame
        imshow("frame", frame)
        if waitKey(1) == ord('q'):
            break

    destroyAllWindows()

