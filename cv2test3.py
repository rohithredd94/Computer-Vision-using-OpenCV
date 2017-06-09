import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('nature.jpg',0)
noise = img.copy()
cv2.imshow('image',img)
cv2.randn(noise,(0),(30))
noise_img = img + noise;
cv2.imshow('Noise-Image',noise_img)

median = cv2.medianBlur(noise_img,5)
cv2.imshow('De-Noised',median)
cv2.waitKey(0)
cv2.destroyAllWindows()