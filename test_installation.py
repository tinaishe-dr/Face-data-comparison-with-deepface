
import sys
import cv2
import deepface
from deepface import DeepFace

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("DeepFace:", deepface.__version__)

print("DeepFace imported successfully!")
