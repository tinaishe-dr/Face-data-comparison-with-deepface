
import cv2

print("OpenCV version:", cv2.__version__)
print("CascadeClassifier available:",
      hasattr(cv2, "CascadeClassifier"))

print("OpenCV location:", cv2.__file__)
