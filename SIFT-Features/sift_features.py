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
    return grad

def get_keypoints(img, draw_keypoints):
    if len(img) > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    corners = cv2.cornerHarris(gray,2,3,0.04) #Opencv Harris-corner detector, faster implementation
    corners = cv2.normalize(corners, corners, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) #Scale the image
    threshold=85
    rows, cols = np.nonzero(corners > threshold)

    #Calculate X and Y gradients
    Ix = calc_grad(gray,3,True,'x')
    Iy = calc_grad(gray,3,True,'y')
    O = np.arctan2(Iy, Ix) #Calculating the orientation aka theta
    Mag = np.sqrt(Ix ** 2 + Iy ** 2) #Calculating the magnitude

    # Assigning the keypoints
    keypoints = np.zeros((len(rows),))
    keypoints = []
    for i in range(len(rows)):
        r = rows[i]; c = cols[i]
        kp = cv2.KeyPoint(c, r, _size=10, _angle=np.rad2deg(O[r,c]), _octave=0)
        keypoints.append(kp)

    # Drawing the keypoints
    if draw_keypoints:
        cv2.drawKeypoints(img, keypoints, img,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img, keypoints

def get_matches(img_A, img_B):
    img_A_feat, kpts1 = get_keypoints(img_A, False)
    img_B_feat, kpts2 = get_keypoints(img_B, False)

    #Using inbuilt SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors_A = sift.compute(img_A, kpts1)[1]
    descriptors_B = sift.compute(img_B, kpts2)[1]

    #Finding matching points
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bfm.match(descriptors_A, descriptors_B)
    matches = sorted(matches, key = lambda x:x.distance)

    return img_A_feat, img_B_feat, kpts1, kpts2, matches

def match_keypoints(img_A, img_B):
    _,_,kpts1, kpts2, matches = get_matches(img_A, img_B)
    matched_image = np.array([])
    matched_image = cv2.drawMatches(img_A, kpts1, img_B, kpts2, matches[:10], flags=2, outImg=matched_image)
    return matched_image

def sift():
    #Part-1: Detect keypoints and draw them on the images
    
    for i, img in enumerate(imgs):
        img = cv2.imread('resources/'+img, cv2.IMREAD_COLOR)

        img, keypoints = get_keypoints(img, True) #Get keypoints
        cv2.imwrite('results/sift-features-detect-'+imgs[i]+'.png', img)

        print("Keypoints detected and drawn on image-"+imgs[i])
    
    #Part-2: Detect keypoints and MATCH them on the images
    for i in range(int (len(imgs)/2)):
        img_A = cv2.imread('resources/'+imgs[2*i], cv2.IMREAD_COLOR)
        img_B = cv2.imread('resources/'+imgs[2*i+1], cv2.IMREAD_COLOR)

        #Get keypoints using OpenCV library and match them using OpenCV SIFT descriptor
        matches = match_keypoints(img_A, img_B)  
        cv2.imwrite('results/sift-features-match-'+imgs[2*i]+','+imgs[2*i+1]+'.png', matches)
        
        print("Keypoints detected and matched on images-"+imgs[2*i]+" & "+imgs[2*i+1])
if __name__ == '__main__':
    print("Executing SIFT feature detection")
    sift()