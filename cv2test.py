import numpy as np
import cv2

img = cv2.imread('nature.jpg')
img = img[0:,0:806,0:]
print(img[0,0,0],img[0,0,1],img[0,0,2])
cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

height, width, channels = img.shape
print(height," ",width," ",channels)

noise = img.copy()
cv2.randn(noise,(0),(1))
noise_img = img + noise * 200
#noise1 = noise * 0.1
cv2.imshow('image1',noise_img) #The image with noise
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
img1 = img[0:,0:403,0:]
img2 = img[0:,403:,0:]

height, width, channels = img1.shape
print(height," ",width," ",channels)
height, width, channels = img2.shape
print(height," ",width," ",channels)

new_img = img1 + img2
#new_img = img * 3
cv2.imshow('new image',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Here python ignores 310th row, 160th column and 1st channel
cropped = img[110:310,10:160,0:1]
cv2.imshow('image',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped = img[110:310,10:160,1:2]
cv2.imshow('image',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped = img[110:310,10:160,2:]
cv2.imshow('image',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''