import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

x_1d = np.arange(3) #No. of rows in sobelX
y_1d = np.arange(3) #No. of columns in sobelY

x,y = np.meshgrid(x_1d,y_1d)

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x,y,sharpen)
plt.show()