import numpy as np
import cv2
'''
Second argument is a flag which specifies the way image should be read.
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively
'''
img = cv2.imread('nature.jpg',0)
img = img[0:,0:806]
cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

height, width= img.shape
print(height," ",width)

noise = img.copy()
cv2.randn(noise,(0),(1)) #Mean = 0 and Variance = 99 with size of noise variable
#cv2.imshow('image1',img)
noise_img = img + noise * 75
#noise1 = noise * 0.1
cv2.imshow('image1',noise_img) #The image with noise
'''
noise2 = noise * 1
cv2.imshow('image2',noise2)
noise3 = noise * 10
cv2.imshow('image3',noise3)
noise4 = noise * 100
cv2.imshow('image4',noise4)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#Genrating Gaussina Noise using numpy arrays
matrix2xN = np.zeros((2,100), np.uint8)
cv2.randn(matrix2xN, 0, 99);
print(matrix2xN);
'''