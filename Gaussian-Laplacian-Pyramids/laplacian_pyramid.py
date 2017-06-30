import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from gaussian_pyramid import *

def laplacian(img, levels = 4,save = False):

    gau_pyr = gaussian(img,levels)
    lap_pyr = [gau_pyr[levels-1]]

    for i in range(levels-1, 0, -1):
        G1 = cv2.pyrUp(gau_pyr[i])
        G2 = gau_pyr[i-1]

        rdiff, cdiff = G2.shape[0] - G1.shape[0], G2.shape[1] - G1.shape[1]
        # if expanded image is bigger then crop it
        if rdiff < 0:
            G1 = G1[:G2.shape[0], :]
            rdiff = 0
        if cdiff < 0:
            G1 = G1[:, :G2.shape[1]]
            cdiff = 0
        # if expanded image is smaller then replicate border
        G1 = cv2.copyMakeBorder(G1, 0, rdiff, 0, cdiff, cv2.BORDER_REPLICATE)

        # subtract the two gaussian levels to get the corresponding laplacian
        L = cv2.subtract(G2, G1)
        lap_pyr.append(L)

    print('Laplacian pyramid complete')
    

    if save:
        save_images(lap_pyr[::-1], 'results/Laplacian-pyramid-'+str(levels)+'-levels.png')
        print('Laplacian Pyramid Images saved')

    return lap_pyr[::-1]

if __name__ == '__main__':
    img = cv2.imread('resources/DataSeq1/yos_img_01.jpg', cv2.IMREAD_COLOR) #First frame of DataSeq1
    laplacian(img, 4, True)