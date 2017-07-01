import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from gaussian_pyramid import *
from lucas_kanade import *


def backwarp(img, flow):
    h, w = flow.shape[:2]
    flow_map = -flow.copy()
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    warped = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
    return warped

def lk_level1(images, levels, winSize):
    gau_pyrs = []
    for i in range(len(images)):
        gau_pyrs += gaussian(images[i], levels)
    
    print('Generating Flows')
    flows = []
    for i in range(len(images)-1):
        flows += [-lk_flow(images[i], images[i+1], winSize)]

    print('Generating Warps')
    warps = []
    for fr, flow in zip(images[1:], flows):
        warps += [backwarp(fr, flow)]

    print('Generating Differences')
    diffs = []
    for i in range(len(warps)):
        diffs += [cv2.subtract(images[i], warps[i])]

    diffs1 = []
    for img in diffs:
        diffs1 += [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)]

    return flows,warps,diffs1

def draw_flows(images, flows, filename):
    #Splitting the images in equal rows and cols
    cols = 2
    rows = np.ceil(float(len(flows)) / cols)

    #Plotting all images
    plt.figure(1)
    for i, (img, flow) in enumerate(zip(images, flows)):
        plot_i = rows * 100 + cols * 10 + 1 + i
        plt.subplot(plot_i)
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        step=int(img.shape[0] / 30)
        x = np.arange(0, img.shape[1], 1)
        y = np.arange(0, img.shape[0], 1)
        x, y = np.meshgrid(x, y)
        plt.quiver(x[::step, ::step], y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='r', pivot='middle', headwidth=5, headlength=5)

    plt.savefig(filename)
    plt.clf()


def warp():
    
    print('Warping Data Sequence 1')
    #Lambda functions in python
    path = lambda i: 'resources/DataSeq1/yos_img_0' + str(i+1) + '.jpg'
    #frs = np.array([cv2.imread(path(i), cv2.IMREAD_GRAYSCALE) for i in range(3)])
    images = np.array([cv2.imread(path(0), cv2.IMREAD_GRAYSCALE)])
    for i in range(1,3):
        img = cv2.imread(path(i), cv2.IMREAD_GRAYSCALE)
        images = np.append(images, np.array([img]), axis=0)

    flows, warps, diffs = lk_level1(images, 1, 15)

    draw_flows(images, flows, 'results/DataSeq1-flowDir.png')
    save_images(diffs, 'results/DataSeq1-difference.png')
    print('Flow and Difference images saved successfully')

    cv2.imwrite('results/DataSeq1-img1-warped.png', warps[0])
    cv2.imwrite('results/DataSeq1-img2-warped.png', warps[1])
    print('Warped Images saved successfully')

    for i, flow in enumerate(flows):
        cv2.imwrite('results/DataSeq1-flows-img'+str(i+1)+'.jpg', draw_flow(flow))
    

    #######***************########
    print('\nWarping Data Sequence 2')
    path = lambda i: 'resources/DataSeq2/' + str(i) + '.png'
    #frs = np.array([cv2.imread(path(i), cv2.IMREAD_GRAYSCALE) for i in range(3)])
    images = np.array([cv2.imread(path(0), cv2.IMREAD_GRAYSCALE)])
    for i in range(1,3):
        img = cv2.imread(path(i), cv2.IMREAD_GRAYSCALE)
        images = np.append(images, np.array([img]), axis=0)

    flows, warps, diffs = lk_level1(images, 3, 7)

    draw_flows(images, flows, 'results/DataSeq2-flowDir.png')
    save_images(diffs, 'results/DataSeq2-difference.png')
    print('Flow and Difference images saved successfully')

    cv2.imwrite('results/DataSeq2-img1-warped.png', warps[0])
    cv2.imwrite('results/DataSeq2-img2-warped.png', warps[1])
    print('Warped Images saved successfully')

    for i, flow in enumerate(flows):
        cv2.imwrite('results/DataSeq2-flows-img'+str(i+1)+'.jpg', draw_flow(flow))




if __name__ == '__main__':
    warp()