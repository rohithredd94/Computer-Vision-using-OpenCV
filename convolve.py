import numpy as np
import cv2
from matplotlib import pyplot as plt
#http://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	pad = int((kW - 1) / 2)
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	#output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 1).astype("uint8")
	return output

image = cv2.imread('COD_template.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
 
# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
 
# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

kernelBank = (
	#("small_blur", smallBlur),
	#("large_blur", largeBlur),
	#("sharpen", sharpen),
	#("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY))
'''
for (kernelName, kernel) in kernelBank:
	print("[INFO] applying {} kernel".format(kernelName))
	convoleOutput = convolve(img, kernel)
	opencvOutput = cv2.filter2D(img, -1, kernel)
	cv2.imshow("original", img)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
'''
kernelName = "Edge"
convoleOutput1 = convolve(img, sobelX)
opencvOutput1 = cv2.filter2D(img, -1, sobelX)

convoleOutput2 = convolve(img, sobelY)
opencvOutput2 = cv2.filter2D(img, -1, sobelY)

convoleOutput = convoleOutput1 + convoleOutput2
opencvOutput = opencvOutput1 + opencvOutput2

cv2.imshow("original", img)
cv2.imshow("{} - convole".format(kernelName), convoleOutput)
cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()