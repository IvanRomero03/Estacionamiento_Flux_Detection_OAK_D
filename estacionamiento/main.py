import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler
import os

ARGS = {
    "MODELS_PATH": os.path.join(os.getcwd(), "models"),
    "YOLO_BLOB": os.path.join(os.getcwd(), "models", "yolov4_tiny.blob"),
    "PLATES_BLOB": os.path.join(os.getcwd(), "models", "license-plate-recognition-barrier-0007.blob"),
}
