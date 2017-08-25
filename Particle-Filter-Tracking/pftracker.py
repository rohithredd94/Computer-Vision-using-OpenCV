import cv2
import numpy as np
import time
import sys
from PF_Tracker import *

videos= ['noisy_debate','pedestrians','pres_debate']
text = ['noisy_debate','pedestrians','pres_debate','pres_debate_hand'] #These contain the model coordinates

def pf_tracker(video, text, frames_to_save=[], play_video=True, num_particles=100, dimensions=2,
                control=10, sim_std=20, alpha=0):
    capture = cv2.VideoCapture('./resources/'+video+'.avi')

    flag, frame = capture.read()
    if not flag:
        print("FATAL ERROR: Unable to read first frame. Program will now exit")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    search_space = np.array(gray.shape)

    f = open('./resources/' + text + '.txt', 'r').read().split()
    s = {'x': float(f[0]), 'y': float(f[1]), 'w': float(f[2]), 'h': float(f[3])} #Extracting the model co-ordinates
    print(s)
    miny = int(s['y']); maxy = int(miny + s['h'])
    minx = int(s['x']); maxx = int(minx + s['w']) #Mapping the co-ordinates onto the frame
    model = gray[miny:maxy, minx:maxx]
    cv2.imwrite('results/firstframe_imagepatch.png', frame[miny:maxy, minx:maxx])

    pftracker = PF_Tracker(model, search_space, num_particles, dimensions, control, sim_std, alpha)

    frame_count = 1
    saved_frames_count = 0
    while capture.isOpened():
        start = time.time()
        flag, frame = capture.read()    
        if not flag:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pftracker.update(gray) #update tracker with new info
        pftracker.visualize_filter(frame) #Draw the new frame

        if len(pftracker.model.shape) < 3:
            color_model = cv2.cvtColor(pftracker.model, cv2.COLOR_GRAY2BGR)
        frame[:model.shape[0], :model.shape[1]] = color_model

        delay = int(25 - (time.time() - start))
        if cv2.waitKey(delay) & 0xFF == ord('q'): #Hit 'q' key at any point to stop video
            break
        cv2.imshow(video, frame)

        if frame_count in frames_to_save:
            cv2.imwrite('results/frame-'+str(frame_count)+'.png', frame)
            saved_frames_count

if __name__ == '__main__':
    print("Particle Filter Tracking Started")
    pf_tracker(videos[2], text[2], [28,84,144], play_video=True, num_particles=100, dimensions=2,
                          control=10, sim_std=20, alpha=0) #Function to start Particle Filter Tracking
    print("Particle Filter Tracking End")

    print("Particle Filter Tracking with appearance model update Started")
    pf_tracker(videos[2], text[3], [28,84,144], play_video=True, num_particles=700, dimensions=2,
                          control=10, sim_std=20, alpha=0) #Function to start Particle Filter Tracking
    print("Particle Filter Tracking with appearance model update End")
