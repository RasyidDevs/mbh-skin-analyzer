import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def annotate_wrinkle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([45, 120, 60])
    upper = np.array([65, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    annotations = []

    for cnt in contours:
        poly = []
        for p in cnt:
            x, y = p[0]
            poly.append([int(x), int(y)])

        annotations.append({"segmentation": poly})

    return annotations


def overlay_annotations(img, annotations):
    img_draw = img.copy()

    for ann in annotations:
        pts = np.array(ann["segmentation"], dtype=np.int32)
        cv2.polylines(img_draw, [pts], False, (0, 255, 0), 2)

    return img_draw

