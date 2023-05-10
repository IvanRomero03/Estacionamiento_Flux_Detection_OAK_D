# This script is intended to be runned on the OAK-D-Lite as a Script Node

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
    


