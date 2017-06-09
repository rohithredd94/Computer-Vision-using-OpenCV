import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
a = np.empty((1,10),np.uint8)
cv2.randn(a,(0),(10))
print(a)
'''
img = cv2.imread('nature.jpg',0)
noise = img.copy()
cv2.randn(noise,(32),(20)) #Mean = 0, Variance = 1
noise_img = img + noise #* 200 #Sigma for noise = 25
cv2.imshow('image',img)
cv2.imshow('noise',noise_img)
#pad = (int) (5 - 1) / 2
#pad = 5
#img = cv2.copyMakeBorder(img, pad, pad, pad, pad,cv2.BORDER_REPLICATE)

blur = cv2.GaussianBlur(img,(11,11),2) #Kernel Size - 11x11, Sigma = 2

cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
'''