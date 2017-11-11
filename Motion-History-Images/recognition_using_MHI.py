import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from frame_differenced_MHI import *


video = lambda a, p, t: 'resources/PS7A' + str(a) + 'P' + str(p) + 'T' + str(t) + '.avi'

def calc_mhis_meis(skip_person_idx=0):

    frames = [70,40,50]
    thetas = [3,3,20]
    taus = [60,40,50]

    mhis = []
    labels = []

    #Triple for loop to run on 3 actions of 3 person with 3 trails
    for action in [1,2,3]:
        for person in [p for p in [1,2,3] if p != skip_person_idx]:
            for trial in [1,2,3]: 
                bin_seq = create_bin_seq(video(action, person, trial), num_frames = frames[person-1],
                        theta = thetas[person-1], blur_ksize=(85,)*2, blur_sigma=0, open_ksize=(9,)*2)

                M_tau = create_mhi_seq(bin_seq, tau = taus[person-1], frames = frames[person-1]).astype(np.float)

                cv2.normalize(M_tau, M_tau, 0.0, 255.0, cv2.NORM_MINMAX)
                mhis.append(M_tau)
                labels.append(action)

    meis = [(255*M > 0).astype(np.uint8) for M in mhis]

    return mhis,meis,labels

def calc_hu_moments(img):
    pq = [[2,0], [0,2], [1,2], [2,1], [2,2], [3,0], [0,3]]
    M_00 = img.sum()
    M_01 = np.sum(np.arange(img.shape[0]).reshape((-1,1)) * img)
    M_10 = np.sum(np.arange(img.shape[1]) * img)
    x_mean = M_10 / M_00
    y_mean = M_01 / M_00

    mu = np.zeros(len(pq))
    eta = np.zeros(len(pq))
    for idx,(p,q) in enumerate(pq):
        cx = (np.arange(img.shape[1]) - x_mean) ** p
        cy = ((np.arange(img.shape[0]) - y_mean) ** q).reshape((-1,1))
        mu[idx] = np.sum(cy * cx * img)
        eta[idx] = mu[idx] / img.sum() ** (1+(p+q)/2)

    return mu, eta

def calc_all_hu_moments(mhis, meis):
    mu_list = []
    eta_list = []

    for mhi, mei in zip(mhis, meis):
        mu1, eta1 = calc_hu_moments(mhi)
        mu2, eta2 = calc_hu_moments(mei)
        mu_list.append(np.append(mu1, mu2))
        eta_list.append(np.append(eta1, eta2))

    return mu_list, eta_list

'''
function borrowed from:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          filename='confusion_matrix.png'):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = (cm * 100 / cm.sum()).astype(np.uint) / 100.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)

def plot_nearest_neighbour_confusion(train_data, labels, filename):
    classifier = cv2.ml.KNearest_create()
    cnf_matrix = np.zeros((3,3))
    np.set_printoptions(precision=2)

    for i in range(len(train_data)):
        
        #Train dataset
        X_train = np.delete(train_data, [i], axis=0)
        y_train = np.delete(labels, [i], axis=0)
        
        #Test dataset
        X_test = np.array([train_data[i]])
        y_test = np.array([labels[i]])
        
        #Train knn classifier
        classifier.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        
        #Predict test data labels
        _,results,_,_ = classifier.findNearest(X_test, 1)
        cnf_matrix[y_test-1, int(results[0])-1] += 1

    plot_confusion_matrix(cnf_matrix, classes=['action1','action2','action3'],
                    normalize=True, title='Confusion matrix', filename=filename)

    return cnf_matrix

def test_confusion_matrix():
    #No arguments since we run on all the videos
    #MHI - Motion History Images, MEI - Motion Energy Images
    mhis, meis, labels = calc_mhis_meis() 

    mu_list, eta_list = calc_all_hu_moments(mhis, meis)

    #Train a k-NN classifier using the hu-moments from both MHIs and MEIs
    train_data = np.array(mu_list).astype(np.float32)
    train_data2 = np.array(eta_list).astype(np.float32)
    labels = np.array(labels).astype(np.int)
    
    plot_nearest_neighbour_confusion(train_data, labels,
                                     'results/confusion-matrix-central-moments.png')

    plot_nearest_neighbour_confusion(train_data2, labels,
                                     'results/confusion-matrix-scaled-moments.png')
    
def final_confusion_matrix():
    cnf_matrices = []
    for person in [1,2,3]:
        #MHI - Motion History Images, MEI - Motion Energy Images
        mhis, meis, labels =calc_mhis_meis(skip_person_idx = person)

        #Calculating Hu Moments
        _,eta_list = calc_all_hu_moments(mhis, meis)

        #Train a k-NN classifier using the hu-moments from both MHIs and MEIs
        train_data = np.array(eta_list).astype(np.float32)
        labels = np.array(labels).astype(np.int)

        cnf_matrix = plot_nearest_neighbour_confusion(train_data, labels, 'results/confusion-matrix-person-'+str(person)+'.png')
        cnf_matrices.append(cnf_matrix)

    cnf_matrix = np.sum(cnf_matrices, 0)
    plot_confusion_matrix(cnf_matrix, ['action1','action2','action3'],
        normalize=True, filename='results/confusion-matrix-average.png')

if __name__ == '__main__':
    test_confusion_matrix() #Testing the confusion matrices

    #Confusion matrices, one for each person and one more which is average of all 3
    final_confusion_matrix()