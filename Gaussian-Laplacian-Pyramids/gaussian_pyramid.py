import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt

def save_images(img, path):
    
        cols = 2
        rows = np.ceil(float(len(img)) / cols)

        plt.figure(1)
        for i, data in enumerate(img):
            plot_i = rows * 100 + cols * 10 + 1 + i
            plt.subplot(plot_i)
            plt.imshow(data)

        plt.savefig(path)  

def gaussian(img, levels = 4, save = False):
    gau_pyr = [img] 
    for i in range(levels-1):
        gau_pyr.append(cv2.pyrDown(gau_pyr[-1])) #gau_pyrDown is the reduce operator in OpenCV

    print('Gaussian pyramid complete')
    if save:
        save_images(gau_pyr, 'results/gaussian-pyramid-'+str(levels)+'-levels.png')

        print('Gaussian Pyramid Images saved')

    return gau_pyr

if __name__ == '__main__':
    img = cv2.imread('resources/DataSeq1/yos_img_01.jpg', cv2.IMREAD_COLOR) #First frame of DataSeq1
    gaussian(img, 4, True)