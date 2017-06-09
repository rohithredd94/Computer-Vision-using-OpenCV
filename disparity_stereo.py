import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read images
L = cv2.imread(os.path.join('input', 'pair2-L.png'), 0) #* (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair2-R.png'), 0) #* (1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
#from disparity_ssd import disparity_ssd
#D_L = disparity_ssd(L, R)
#D_R = disparity_ssd(R, L
plt.figure(1);
plt.imshow(L,'gray');
plt.title('Left-Original')
plt.figure(2);
plt.imshow(R, 'gray');
plt.title('Right-Original')
stereo = cv2.StereoBM_create(numDisparities = 96, blockSize=17)
disparity = stereo.compute(L,R);
plt.figure(3);
plt.imshow(disparity, 'gray')
plt.title('Disparity-Map-L&R')
'''
R and L doesnt work very good for some reason
stereo = cv2.StereoBM_create(numDisparities = 192, blockSize=7)
disparity1 = stereo.compute(R,L);
plt.figure(4);
plt.imshow(disparity1, 'gray')
plt.title('Disparity-Map-R&L')
'''
plt.draw();
plt.pause(1) # <-------
input("<Hit Enter To Close>")
plt.close("all")
