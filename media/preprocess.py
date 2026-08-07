import cv2
from pathlib import Path
import math

def preprocess(image_path: str):
    """Black out noise region"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # height = 480, width = 854
    print(height, width)
    x, y, w, h = 0, int(height*0.90625), width, int(height*0.052)
    # x, y, w, h = 0, 435, width, 25
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    x, y, w, h = int(width*0.8138),int(height*0.0729), int(width*0.1054),int(height*0.084)
    # x, y, w, h = 695,35, 90,40
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image
