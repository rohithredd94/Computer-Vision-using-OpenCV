import cv2
import numpy as np
import sys
import random
from collections import OrderedDict

imgs = ['transA.jpg', 'transB.jpg', 'simA.jpg', 'simB.jpg']

def calc_grad(img, k_sobel, norm,k):
    if k == 'x':
        grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, k_sobel)
    elif k == 'y':
        grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, k_sobel)

    if norm:
        grad = cv2.normalize(grad, grad, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #grad = grad
    return grad

def harris_values(img, window_size, harris_scoring, norm):
    #Calculate X and Y gradients
    grad_x = calc_grad(img, 3, False, 'x')
    grad_y = calc_grad(img, 3, False, 'y')
    grad_xx = grad_x ** 2
    grad_xy = grad_x * grad_y
    grad_yx = grad_y * grad_x
    grad_yy = grad_y ** 2

    #Calculate the weight window matrix
    c = np.zeros((window_size,)*2, dtype=np.float32);   
    c[int (window_size / 2), int (window_size / 2)] = 1.0
    w = cv2.GaussianBlur(c, (window_size,)*2, 0)

    #Calculating the harris window values for the given image
    har_val = np.zeros(img.shape, dtype=np.float32)
    for r in range(int (w.shape[0]/2), int (img.shape[0] - w.shape[0]/2)): #Iterating over the window size
        print(r)
        minr = int (max(0, r - w.shape[0]/2))
        maxr = int (min(img.shape[0], minr + w.shape[0]))
        for c in range(int (w.shape[1]/2), int (img.shape[1] - w.shape[1]/2)):
            minc = int (max(0, c - w.shape[1]/2))
            maxc = int (min(img.shape[1], minc + w.shape[1]))
            wgrad_xx = grad_xx[minr:maxr, minc:maxc]
            wgrad_xy = grad_xy[minr:maxr, minc:maxc]
            wgrad_yx = grad_yx[minr:maxr, minc:maxc]
            wgrad_yy = grad_yy[minr:maxr, minc:maxc]
            m_xx = (w * wgrad_xx).sum()
            m_xy = (w * wgrad_xy).sum()
            m_yx = (w * wgrad_yx).sum()
            m_yy = (w * wgrad_yy).sum()
            M = np.array([m_xx, m_xy, m_yx, m_yy]).reshape((2,2))
            har_val[r,c] = np.linalg.det(M)- harris_scoring * (M.trace() ** 2)
    #Scaling the images
    if norm:
        har_val = cv2.normalize(har_val, har_val, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return har_val

def harris_corners(img, window_size, harris_scoring, threshold, nms_size):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate harris values for all valid pixels
    corners = harris_values(img, window_size, harris_scoring, False)
    # apply thresholding
    corners = corners * (corners > (threshold * corners.max())) * (corners > 0)
    # apply non maximal suppression
    rows, columns = np.nonzero(corners)
    new_corners = np.zeros(corners.shape)
    for r,c in zip(rows,columns):
        minr = int (max(0, r - nms_size / 2))
        maxr = int (min(img.shape[0], minr + nms_size))
        minc = int (max(0, c - nms_size / 2))
        maxc = int (min(img.shape[1], minc + nms_size))
        if corners[r,c] == corners[minr:maxr,minc:maxc].max():
            new_corners[r,c] = corners[r,c]
            #  corners[minr:r, minc:c] = 0
            #  corners[r+1:maxr, c+1:maxc] = 0
    return new_corners

def harris():
    images = imgs[0:3:2];
    for i, img in enumerate(images):
        img = cv2.imread('resources/'+img, cv2.IMREAD_GRAYSCALE)#Read the image
        #Calculate X and Y gradients
        print("Calculating X & Y Gradients")
        img_grad_x = calc_grad(img, 3, True, 'x')
        img_grad_y = calc_grad(img, 3, True, 'y')
        #Save the calculated gradients
        cv2.imwrite('results/gradients-'+images[i]+'.png', np.hstack((img_grad_x, img_grad_y)))
        print("X & Y Gradients are saved to images in results folder")

    #Read the images and calculate harris values for all the images
    for i, img in enumerate(imgs):
        img = cv2.imread('resources/'+img, cv2.IMREAD_GRAYSCALE)#Read the image
        n = i
        har_val = harris_values(img, 3, 0.04, True)
        cv2.imwrite('results/harris-values'+imgs[i]+'.png', har_val)
        print("Harris Values are saved to images in results folder")

        img = np.float32(img)
        corners = harris_corners(img, 3, 0.04,1e-3, 5)
        x = img.shape[0]
        y = img.shape[1]
        for i in range(x):
            for j in range(y):
                if(corners[i][j] > 0):
                    img[i][j] = 255
        cv2.imwrite('results/harris-corners'+imgs[n]+'.png', img)
        print("Harris Corners are saved to images in results folder")

if __name__ == '__main__':
    print("Executing Harris Corners")
    harris()
