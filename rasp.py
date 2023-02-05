import imutils
import cv2
import socket
import pickle
import numpy as np
import time
import hashlib
import requests
import json

import signal

from imutils.object_detection import non_max_suppression
from dotenv import dotenv_values

config = dotenv_values(".env")

def handler(signum, frame):
    ret = False


# Init signal for close
signal.signal(signal.SIGINT, handler)

# Init hog detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start socket 
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('192.168.1.23', 8100))
url = 'http://' + config["ADRESSE_SOCKET"] + ':' + config['PORT_SOCKET'] + '/upload'

# Init camera
cam = cv2.VideoCapture(0)
if cam.isOpened() == False:
    print("Error")


nb_images_send = 0
ret = True
previous_time = time.time()
while ret:
    current_time = time.time()
    dt = current_time - previous_time
    previous_time = current_time

    print(1/dt)
    
    # Image
    ret, image = cam.read()
    try:
        ratio = image.shape[1] / 400
    except:
        break
    orig_image = image.copy()

    # Resize image
    # image = imutils.resize(image, width=min(500, image.shape[1]))
    # image = imutils.resize(image, width=int(image.shape[1] / 2))

    # Person detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(32, 32), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    people_positions = []
    for (xA, yA, xB, yB) in pick:
        people_positions.append((int(xA), int(yA), int(xB), int(yB)))

    if people_positions:
        print(people_positions)
        # Encode image
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        # result, image_bytes = cv2.imencode('.jpg', orig_image, encode_param)
        _, jpg_image = cv2.imencode('.jpg', orig_image)

        # Define informations
        informations = {"people_position": people_positions, "timestamp": time.time()}

        # Concatenate informations and image
        # data = pickle.dumps((informations, image_bytes))

        # length = len(data)
        # checksum = hashlib.sha256(data).hexdigest()

        # print(f'lehgth: {length} | checksum: {checksum} | people: {people_positions}')

        data = {
            'people_positions': json.dumps(people_positions),
            'kilometre': config['KILOMETRE']
        }

        response = requests.post(url, data=data, files={'image': ('image.jpg', jpg_image.tobytes(), 'image/jpeg')})

        if response.status_code != 200:
            print('Error...')
        # client_socket.send(length.to_bytes(4, byteorder='big'))
        # client_socket.send(checksum.encode())

        # # Send image and informations
        # client_socket.sendall(data)
        nb_images_send += 1

# client_socket.send('FIN'.encode('utf-8'))

# data
print(f'number image send: {nb_images_send}')

# Close socket
# client_socket.close()