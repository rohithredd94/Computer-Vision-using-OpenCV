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
    img_A_feat, kptsA = get_keypoints(img_A, False)
    img_B_feat, kptsB = get_keypoints(img_B, False)

    #Using inbuilt SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors_A = sift.compute(img_A, kptsA)[1]
    descriptors_B = sift.compute(img_B, kptsB)[1]

    #Finding matching points
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bfm.match(descriptors_A, descriptors_B)
    matches = sorted(matches, key = lambda x:x.distance)

    return img_A_feat, img_B_feat, kptsA, kptsB, matches

def ransac_trans(imgA,imgB):
    imgA, imgB, kptsA, kptsB, matches = get_matches(imgA, imgB)
    tolerance = 10; best_match = 0; consensus_set = []

    for i in range(100):
        idx = random.randint(0, len(matches)-1)
        kpA = kptsA[matches[idx].queryIdx]
        kpB = kptsB[matches[idx].trainIdx]
        dx = int(kpA.pt[0] - kpB.pt[0])
        dy = int(kpA.pt[1] - kpB.pt[1])
        temp_consensus_set = []
        for j, match in enumerate(matches):
            kpA = kptsA[match.queryIdx]
            kpB = kptsB[match.trainIdx]
            dxi = int(kpA.pt[0] - kpB.pt[0])
            dyi = int(kpA.pt[1] - kpB.pt[1])
            if abs(dx - dxi) < tolerance and abs(dy - dyi) < tolerance:
                temp_consensus_set.append(j)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_match = idx

    #Calculating best match translation
    kpA = kptsA[matches[best_match].queryIdx]
    kpB = kptsB[matches[best_match].trainIdx]
    dx = int(kpA.pt[0] - kpB.pt[0])
    dy = int(kpA.pt[1] - kpB.pt[1])
    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])

    #Drawing matches with biggest consensus
    matched_image = cv2.drawMatches(imgA, kptsA, imgB, kptsB, consensus_matches, flags=2, outImg=matched_image)
    print('Best match: idx = %d with consensus = %d or %d%%\nTranslation: dx=%dpx and dy=%dpx'%(best_match, len(consensus_set), 100*len(consensus_set) / len(matches), dx, dy))
    return matched_image

def ransac_sim(simA, simB):
    _, _, kptsA, kptsB, matches = get_matches(simA, simB)
    imgA = simA; imgB = simB
    tolerance = 1 #Used to compare similarity matrices of match pairs
    consensus_set = []; best_sim = []

    #Find consensus of translation between a random keypoint and the rest for a number of times to find the best match regarding translation
    for i in range(100):
        idxs = random.sample(range(len(matches)), 2)

        #Calculating similarity between kp1 and kp2
        kp11 = kptsA[matches[idxs[0]].queryIdx]
        kp12 = kptsB[matches[idxs[0]].trainIdx]
        kp21 = kptsA[matches[idxs[1]].queryIdx]
        kp22 = kptsB[matches[idxs[1]].trainIdx]
        A = np.array([[kp11.pt[0], -kp11.pt[1], 1, 0],
                      [kp11.pt[1], kp11.pt[0], 0, 1],
                      [kp21.pt[0], -kp21.pt[1], 1, 0],
                      [kp21.pt[1], kp21.pt[0], 0, 1]])
        b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1]])
        sim,_,_,_ = np.linalg.lstsq(A, b)
        sim = np.array([[sim[0], -sim[1], sim[2]],
                        [sim[1], sim[0], sim[3]]])
        temp_consensus_set = []
        for j in range(len(matches) - 1):
            match = matches[j]
            kp11 = kptsA[matches[j].queryIdx]
            kp12 = kptsB[matches[j].trainIdx]
            kp21 = kptsA[matches[j+1].queryIdx]
            kp22 = kptsB[matches[j+1].trainIdx]
            A = np.array([[kp11.pt[0], -kp11.pt[1], 1, 0],
                          [kp11.pt[1], kp11.pt[0], 0, 1],
                          [kp21.pt[0], -kp21.pt[1], 1, 0],
                          [kp21.pt[1], kp21.pt[0], 0, 1]])
            b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1]])
            sim2,_,_,_ = np.linalg.lstsq(A, b)
            sim2 = np.array([[sim2[0], -sim2[1], sim2[2]],
                             [sim2[1], sim2[0], sim2[3]]])
            if (np.array(np.abs(sim-sim2)) < tolerance).all():
                temp_consensus_set.append(j)
                temp_consensus_set.append(j+1)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_sim = sim

    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])

    #Drawing matches with biggest consensus
    matched_image = cv2.drawMatches(imgA, kptsA, imgB, kptsB, consensus_matches[:100],flags=2, outImg=matched_image)
    print('Best match:\nsim = \n%s\n with consensus = %d or %d%%'%(best_sim, len(consensus_set)/2, 100*len(consensus_set)/2/len(matches)))
    return matched_image, best_sim

def ransac_sim_affine(simA, simB):
    imgA, imgB, kptsA, kptsB, matches = get_matches(simA, simB)
    tolerance = 1 #Used to compare similarity matrices of match pairs
    consensus_set = []; best_sim = []

    for i in range(100):
        idxs = random.sample(range(len(matches)), 3)

        #Calculating similarity between kp1, kp2 and kp3
        kp11 = kptsA[matches[idxs[0]].queryIdx]
        kp12 = kptsB[matches[idxs[0]].trainIdx]
        kp21 = kptsA[matches[idxs[1]].queryIdx]
        kp22 = kptsB[matches[idxs[1]].trainIdx]
        kp31 = kptsA[matches[idxs[2]].queryIdx]
        kp32 = kptsB[matches[idxs[2]].trainIdx]

        ptsA = np.float32([[kp11.pt[0], kp11.pt[1]],
                           [kp21.pt[0], kp21.pt[1]],
                           [kp31.pt[0], kp31.pt[1]]])
        ptsB = np.float32([[kp12.pt[0], kp12.pt[1]],
                           [kp22.pt[0], kp22.pt[1]],
                           [kp32.pt[0], kp32.pt[1]]])
        Sim = cv2.getAffineTransform(ptsA, ptsB)
        temp_consensus_set = []
        for j in range(len(matches)-2):
            kp11 = kptsA[matches[j].queryIdx]
            kp12 = kptsB[matches[j].trainIdx]
            kp21 = kptsA[matches[j+1].queryIdx]
            kp22 = kptsB[matches[j+1].trainIdx]
            kp31 = kptsA[matches[j+2].queryIdx]
            kp32 = kptsB[matches[j+2].trainIdx]
            ptsA = np.float32([[kp11.pt[0], kp11.pt[1]],
                               [kp21.pt[0], kp21.pt[1]],
                               [kp31.pt[0], kp31.pt[1]]])
            ptsB = np.float32([[kp12.pt[0], kp12.pt[1]],
                               [kp22.pt[0], kp22.pt[1]],
                               [kp32.pt[0], kp32.pt[1]]])
            Sim2 = cv2.getAffineTransform(ptsA, ptsB)
            if (np.array(np.abs(Sim-Sim2)) < tolerance).all():
                temp_consensus_set.append(j)
                temp_consensus_set.append(j+1)
                temp_consensus_set.append(j+2)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_sim = Sim

    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])

    #Drawing matches with biggest consensus
    matched_image = cv2.drawMatches(imgA, kptsA, imgB, kptsB, consensus_matches, flags=2, outImg=matched_image)
    print('Best match:\nSim=\n%s\n with consensus=%d or %d%%'%(best_sim, len(consensus_set)/3, 100*len(consensus_set)/len(matches)))
    return matched_image, best_sim

def warp(simA, simB, transform):
    warpedB = cv2.warpAffine(simB, transform, simB.shape[1::-1], flags=cv2.WARP_INVERSE_MAP)
    blend = warpedB * 0.5
    blend[:simA.shape[0], :simA.shape[1]] += simA * 0.5
    blend = cv2.normalize(blend, blend, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return blend

def ransac():
    print("Matching image set - trans")
    transA = cv2.imread('resources/transA.jpg', cv2.IMREAD_COLOR)
    transB = cv2.imread('resources/transB.jpg', cv2.IMREAD_COLOR)
    matches = ransac_trans(transA, transB)
    cv2.imwrite('results/ransac-trans.png', matches)
    print("RANSAC Matching successful for 'trans' image set\n")

    print("Matching image set - sim")
    simA = cv2.imread('resources/simA.jpg', cv2.IMREAD_COLOR)
    simB = cv2.imread('resources/simB.jpg', cv2.IMREAD_COLOR)
    matches,transform = ransac_sim(simA, simB)
    cv2.imwrite('results/ransac-sim.png', matches)
    print("RANSAC Matching successful for 'sim' image set\n")

    print("Affine Matching image set - sim")
    matches,transform_affine = ransac_sim_affine(simA, simB)
    cv2.imwrite('results/ransac-sim-affine.png', matches)
    print("RANSAC affine Matching successful for 'sim' image set\n")

    print("Warping image set - sim")
    if len(transform) == 0:
        print("RANSAC image warping failed for 'sim' image set")
    else:
        blend = warp(simA, simB, transform)
        cv2.imwrite('results/ransac-sim-warped.png', blend)
        print("RANSAC image warping successful for 'sim' image set\n")

    print("Affine Warping image set - sim")
    if len(transform_affine) == 0:
        print("RANSAC affine image warping failed for 'sim' image set")
    else:
        blend = warp(simA, simB, transform_affine)
        cv2.imwrite('results/ransac-sim-warped-affine.png', blend)
        print("RANSAC affine image warping successful for 'sim' image set\n")

if __name__ == '__main__':
    print("Executing RANSAC Feature detection and matching\n")
    ransac()