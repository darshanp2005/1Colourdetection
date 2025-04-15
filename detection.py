# For Colour detection, we will use HSV (Hue ,Saturation, Value) Colour Space
import cv2
from PIL import Image  # Pillow Library
from util import get_limits

yellow = [0, 255, 255]  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)  # to get mask from all , color to detect

    # Using Contours for better bounding box 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)  # for Boundbox

    print(bbox)  # if yellow than it will give dimensions

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
