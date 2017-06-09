import numpy as np
import cv2

img = cv2.imread('laptop.jpg')
img1 = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow("Edges",edges)

#Normal Hough Transform
lines = cv2.HoughLines(edges,1,np.pi/180,170)
print(lines.shape[0])

for i in range(lines.shape[0]):
	r = lines[i][0][0]
	t = lines[i][0][1]
	a = np.cos(t)
	b = np.sin(t)
	x0 = a*r
	y0 = b*r
	x1 = int(x0 - 1000*b)
	y1 = int(y0 + 1000*a)
	x2 = int(x0 + 1000*b)
	y2 = int(y0 - 1000*a)
	cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
cv2.imshow("Lines",img)


#Probabilistic Hough Transform
minLineL = 10
maxLineG = 10
linesP = cv2.HoughLinesP(edges, 1, np.pi/180,75, minLineL, maxLineG)
print(linesP.shape)

for i in range(linesP.shape[0]):
	xp1 = linesP[i][0][0]
	yp1 = linesP[i][0][1]
	xp2 = linesP[i][0][2]
	yp2 = linesP[i][0][3]
	cv2.line(img1, (xp1,yp1),(xp2,yp2),(0,255,0),2)
	#print("Hello")
cv2.imshow("Lines_P",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()