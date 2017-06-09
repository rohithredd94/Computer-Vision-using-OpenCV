import numpy as np
import imutils
import glob
import cv2

image = cv2.imread('COD_template.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 50, 200)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()