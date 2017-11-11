import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

video = lambda a, p, t: 'resources/PS7A' + str(a) + 'P' + str(p) + 'T' + str(t) + '.avi'

def create_bin_seq(video, num_frames=10, theta=127, blur_ksize=(3,3),
    blur_sigma=1, open_ksize = (3,3)):
    
    capture = cv2.VideoCapture(video)
    bin_seq = []
    ret, frame = capture.read()

    if not ret:
        print('Failed to retrive frame: Bad/No video file')
        exit(0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, blur_ksize, blur_sigma) #Blurring to get a cleaner final image
    kernel = np.ones(open_ksize, dtype=np.uint8)

    for i in range(num_frames):
        ret, new_frame = capture.read()

        if not ret:
            print('Failed to retrive frame: Bad/No video file')
            exit(0)

        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.GaussianBlur(new_frame, blur_ksize, blur_sigma)

        #Checking to find if difference is above threshold or not
        bin_frame = np.abs(cv2.subtract(new_frame, frame)) >= theta
        bin_frame = bin_frame.astype(np.uint8)

        #OpenCV's morphological OPEN operator
        bin_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_OPEN, kernel)

        #Append every frame's binary equivalent to final output
        bin_seq.append(bin_frame)
        frame = new_frame

    return bin_seq

def create_mhi_seq(bin_seq, tau = 0.5, frames = 30):
    M_tau = np.zeros(bin_seq[0].shape, dtype=np.float)

    for f, B_tau in enumerate(bin_seq):
        M_tau = tau * (B_tau == 1) + np.clip(np.subtract(M_tau, np.ones(M_tau.shape)), 0, 255) * (B_tau == 0)

        if(f == frames):
            break

    return M_tau.astype(np.uint8)


def construct_binary_sequence():
    #Video files available are (1,1,1), (1,1,2), (1,1,3)
    bin_seq = create_bin_seq(video(1,1,1), num_frames = 31, theta = 2,
                blur_ksize=(55,)*2, blur_sigma = 0, open_ksize=(9,)*2)

    for i in [10,20,30]:
        seq = bin_seq[i]
        cv2.normalize(seq,seq,0,255,cv2.NORM_MINMAX)
        cv2.imwrite('results/binary-sequence-frame-'+str(i)+'.png',seq)

def construct_mhi_sequence():
    frames = [35,30,30]
    thetas = [4,4,4]
    taus = [40,35,30]

    for i in range(3):
        #Video files available are (1,2,1), (1,2,2), (1,2,3)
        bin_seq = create_bin_seq(video(1,2,i+1), frames[i], thetas[i], 
                    blur_ksize=(85,)*2, blur_sigma=0, open_ksize=(9,)*2)

        #For better MHI comparision, modify create_bin_seq to take frames in between instead
        #of only the first few frames. The code now takes only the first few frames of the video
        #The logic nevertheless will remain same
        M_tau = create_mhi_seq(bin_seq, taus[i], frames[i]).astype(np.float)

        cv2.normalize(M_tau, M_tau, 0.0, 255.0, cv2.NORM_MINMAX)
        cv2.imwrite('results/mhi-sequence-video-'+str(i+1)+'.png',M_tau)

if __name__ == '__main__':
    construct_binary_sequence() #Just build the binary sequence
    construct_mhi_sequence() #Build Motion History Images