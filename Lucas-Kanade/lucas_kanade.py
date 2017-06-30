import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt


def lk_flow(img1,img2, k): #Windows size - k
    
    #Initializing gradient matrices for 3 dimensions
    gx = np.zeros(img1.shape, dtype=np.float32)
    gy = np.zeros(img1.shape, dtype=np.float32)
    gt = np.zeros(img1.shape, dtype=np.float32)

    #Calculating gradients
    gx[1:-1, 1:-1] = cv2.subtract(img1[1:-1, 2:], img1[1:-1, :-2]) / 2
    gy[1:-1, 1:-1] = cv2.subtract(img1[2:, 1:-1], img1[:-2, 1:-1]) / 2
    gt[1:-1, 1:-1] = cv2.subtract(img1[1:-1, 1:-1], img2[1:-1, 1:-1])

    specs = np.zeros(img1.shape + (5,))
    specs[..., 0] = gx ** 2
    specs[..., 1] = gy ** 2
    specs[..., 2] = gx * gy
    specs[..., 3] = gx * gt
    specs[..., 4] = gy * gt

    specs = np.zeros(img1.shape + (5,))
    specs[..., 0] = gx ** 2
    specs[..., 1] = gy ** 2
    specs[..., 2] = gx * gy
    specs[..., 3] = gx * gt
    specs[..., 4] = gy * gt
    del gt, gx, gy
    cum_specs = np.cumsum(np.cumsum(specs, axis=0), axis=1)
    del specs
    k_specs = (cum_specs[2 * k + 1:, 2 * k + 1:] -
                  cum_specs[2 * k + 1:, :-1 - 2 * k] -
                  cum_specs[:-1 - 2 * k, 2 * k + 1:] +
                  cum_specs[:-1 - 2 * k, :-1 - 2 * k])
    del cum_specs

    flow = np.zeros(img1.shape + (2,))
    det = k_specs[...,0] * k_specs[..., 1] - k_specs[..., 2] **2

    flow_x = np.where(det != 0,
                     (k_specs[..., 1] * k_specs[..., 3] -
                      k_specs[..., 2] * k_specs[..., 4]) / det,
                      0)
    flow_y = np.where(det != 0,
                     (k_specs[..., 0] * k_specs[..., 4] -
                      k_specs[..., 2] * k_specs[..., 3]) / det,
                      0)
    flow[k + 1: -1 - k, k + 1: -1 - k, 0] = flow_x[:-1, :-1]
    flow[k + 1: -1 - k, k + 1: -1 - k, 1] = flow_y[:-1, :-1]
    flow = flow.astype(np.float32)
    return flow

#Simple implementation but not so good at performance
def lk_flow_alternate(frame1, frame2, filter_size=5):
    gt = cv2.subtract(frame1, frame2).astype(np.float32)
    gx, gy = np.gradient(frame1)
    gx = gx.astype(np.float32)
    gx = gx.astype(np.float32)

    #Calculating the Gradient Product accumulations for each pixel
    gxx = cv2.boxFilter(gx**2, -1, ksize=(filter_size,)*2, normalize=True)
    gxy = cv2.boxFilter(gx*gy, -1, ksize=(filter_size,)*2, normalize=True)
    gyy = cv2.boxFilter(gy**2, -1, ksize=(filter_size,)*2, normalize=True)
    gxt = cv2.boxFilter(gx*gt, -1, ksize=(filter_size,)*2, normalize=True)
    gyt = cv2.boxFilter(gy*gt, -1, ksize=(filter_size,)*2, normalize=True)
    del gt, gx, gy

    #Calculating the Displacement Matrices U, V
    rows, cols = frame1.shape
    flow = np.zeros((rows, cols, 2), dtype=np.float32)
    A = np.dstack((gxx, gxy, gxy, gyy))
    b = np.dstack((-gxt, -gyt))
    for r in range(rows):
        for c in range(cols):
            flow[r,c,:] = np.linalg.lstsq(A[r,c].reshape((2,2)), b[r,c])[0]

    return -flow


def draw_flow_on_original(img, flow, filename):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)

    #Creating a grid
    x, y = np.meshgrid(x, y)
    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = int(img.shape[0] / 50)
    plt.quiver(x[::step, ::step], y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='r', pivot='middle', headwidth=2, headlength=3)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

def draw_flow(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2]+(3,), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis

def lk():
    print('Testing on easy images - 2-5 pixel shift')
    img1 = cv2.imread('resources/TestSeq/Shift0.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('resources/TestSeq/ShiftR2.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('resources/TestSeq/ShiftR5U5.png', cv2.IMREAD_GRAYSCALE)

    flow12 = lk_flow(img1, img2, 15)
    flow13 = lk_flow(img1, img3, 15)
    print('Flow Calculation Complete')
    #flow12 = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #flow13 = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    draw_flow_on_original(img1, flow12, 'results/flowdir1-2.png')
    draw_flow_on_original(img1, flow13, 'results/flowdir1-3.png')
    print('Flow direction images saved successfully')

    cv2.imwrite('results/flow1-2.jpg', draw_flow(flow12))
    cv2.imwrite('results/flow1-3.jpg', draw_flow(flow13))
    print('Raw Flow images saved successfully\n')

    print('Testing on hard images - 10-40 pixel shift')
    img4 = cv2.imread('resources/TestSeq/ShiftR10.png', cv2.IMREAD_GRAYSCALE)
    img5 = cv2.imread('resources/TestSeq/ShiftR20.png', cv2.IMREAD_GRAYSCALE)
    img6 = cv2.imread('resources/TestSeq/ShiftR40.png', cv2.IMREAD_GRAYSCALE)

    flow14 = lk_flow(img1, img4, 15)
    flow15 = lk_flow(img1, img5, 15)
    flow16 = lk_flow(img1, img6, 15)
    print('Flow Calculation Complete')

    draw_flow_on_original(img1, flow14, 'results/flowdir1-4.png')
    draw_flow_on_original(img1, flow15, 'results/flowdir1-5.png')
    draw_flow_on_original(img1, flow16, 'results/flowdir1-6.png')
    print('Flow direction images saved successfully')

    cv2.imwrite('results/flow1-4.jpg', draw_flow(flow14))
    cv2.imwrite('results/flow1-5.jpg', draw_flow(flow15))
    cv2.imwrite('results/flow1-6.jpg', draw_flow(flow16))
    print('Raw Flow images saved successfully\n')


if __name__ == '__main__':
    print('Executing Lucas Kanade optic flow\n')
    lk()