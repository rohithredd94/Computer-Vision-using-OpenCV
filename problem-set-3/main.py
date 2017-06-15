import cv2
import numpy as np
import sys
import random
from collections import OrderedDict

M_norm_a = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)
def load_file(name):
    temp = []
    f = open(name, 'r')
    for line in f:
        temp.append(line.strip().split())
    f.close()
    return np.array(temp, dtype=np.float32)

def least_squares_M_solver(pts2d, pts3d):
    #  M = np.zeros((12,1), dtype=np.float32)
    num_pts = pts2d.shape[0]
    A = np.zeros((2*num_pts,11), dtype=np.float32)
    b = np.zeros(2*num_pts, dtype=np.float32)
    x = pts2d[:,0]
    y = pts2d[:,1]
    X = pts3d[:,0]
    Y = pts3d[:,1]
    Z = pts3d[:,2]
    #  for i in range(num_pts):
    zeros = np.zeros(num_pts)
    ones = np.ones(num_pts)
    A[::2,:]   = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z))
    A[1::2,:] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z))
    b[::2] = x
    b[1::2] = y
    M,res,_,_ = np.linalg.lstsq(A, b)
    M = np.append(M, 1)
    M = M.reshape((3,4))
    return M, res

def svd_M_solver(pts2d, pts3d):
    #  M = np.zeros((12,1), dtype=np.float32)
    num_pts = pts2d.shape[0]
    A = np.zeros((2*num_pts,12), dtype=np.float32)
    b = np.zeros(2*num_pts, dtype=np.float32)
    x = pts2d[:,0]
    y = pts2d[:,1]
    X = pts3d[:,0]
    Y = pts3d[:,1]
    Z = pts3d[:,2]
    zeros = np.zeros(num_pts)
    ones = np.ones(num_pts)
    A[::2,:]   = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z, -x))
    A[1::2,:] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z, -y))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    M = V.T[:,-1]
    M = M.reshape((3,4))
    return M

def calc_residual(pts2d, pts3d, M):
    pts2d_proj = np.array([np.dot(M, np.append(pt_3d,1)) for pt_3d in pts3d])
    pts2d_proj = pts2d_proj[:,:2] / pts2d_proj[:,2:]#.reshape(4,1)
    res = np.linalg.norm(pts2d - pts2d_proj)
    return res

def best_M(pts2d, pts3d, calibpts, testpts, iter):
    num_pts = pts2d.shape[0]
    M = np.zeros((3,4), dtype=np.float32)
    res = 1e9
    for iter in range(iter):
        idxs = random.sample(range(num_pts), calibpts)
        M_tmp,_ = least_squares_M_solver(pts2d[idxs], pts3d[idxs])
        #  M_tmp = svd_M_solver(pts2d[idxs], pts3d[idxs])
        test_idxs = [i for i in range(num_pts) if i not in idxs]
        test_idxs = random.sample(test_idxs, testpts)
        res_tmp = calc_residual(pts2d[test_idxs], pts3d[test_idxs], M_tmp)
        if res_tmp < res:
            res = res_tmp
            M = M_tmp
    return M, res

def least_squares_F_solver(a_pts, b_pts):
    num_pts = a_pts.shape[0]
    ua = a_pts[:,0]
    va = a_pts[:,1]
    ub = b_pts[:,0]
    vb = b_pts[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va))
    b = -np.ones(num_pts)
    F = np.linalg.lstsq(A, b)[0]
    F = np.append(F, 1)
    F = F.reshape((3,3))
    return F

def fun_mat_rank2(F):
    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    S = np.diag(S)
    F = np.dot(np.dot(U,S), V)
    return F

def draw_eplines(img_a,img_b,F,a_pts,b_pts): #Find and draw epipolar lines
    lines_a = np.dot(F.T, b_pts.T).T
    lines_b = np.dot(F, a_pts.T).T
    n, m, _ = img_a.shape
    line_L = np.cross([0,0,1],[n,0,1])
    line_R = np.cross([0,m,1],[n,m,1])

    for line_a, line_b in zip(lines_a, lines_b):
        P_a_L = np.cross(line_a, line_L)
        P_a_R = np.cross(line_a, line_R)
        P_a_L = (P_a_L[:2] / P_a_R[2]).astype(int)
        P_a_R = (P_a_R[:2] / P_a_R[2]).astype(int)
        cv2.line(img_a, tuple(P_a_L[:2]), tuple(P_a_R[:2]), (0,255,0), thickness=2)
        P_b_L = np.cross(line_b, line_L)
        P_b_R = np.cross(line_b, line_R)
        P_b_L = (P_b_L[:2] / P_b_R[2]).astype(int)
        P_b_R = (P_b_R[:2] / P_b_R[2]).astype(int)
        cv2.line(img_b, tuple(P_b_L[:2]), tuple(P_b_R[:2]), (0,255,0), thickness=2)
    return img_a,img_b

def task_1(): #Estimating Camera Projection Matrix
    print("Executing task: 1 \n==================")
    pts2d_norm = load_file('resources/pts2d-norm-pic_a.txt')
    pts3d_norm = load_file("resources/pts3d-norm.txt")
    M, res = least_squares_M_solver(pts2d_norm, pts3d_norm) #Finding M-Project matrix using LS method
    pts2d_proj = np.dot(M, np.append(pts3d_norm[-1],1))
    pts2d_proj = pts2d_proj[:2] / pts2d_proj[2]
    res = np.linalg.norm(pts2d_norm[-1] - pts2d_proj)
    print("Projection Matrix using Least Squares Method")
    print("M (Projection Matrix) =\n%s"%M)
    print("Point %s projected to point %s"%(pts2d_norm[-1] ,pts2d_proj))
    print("Residual: %.4f\n"%res)

    M = svd_M_solver(pts2d_norm, pts3d_norm) #Finding M-Project matrix using SVD method
    pts2d_proj = np.dot(M, np.append(pts3d_norm[-1],1))
    pts2d_proj = pts2d_proj[:2] / pts2d_proj[2]
    res = np.linalg.norm(pts2d_norm[-1] - pts2d_proj)
    print("Projection Matrix using SVD Method")
    print("M (Projection Matrix) =\n%s"%M)
    print("Point %s projected to point %s"%(pts2d_norm[-1] ,pts2d_proj))
    print("Residual: %.4f\n"%res)

    #Finding the best the best projection matrix using varying residuals
    pts2d = load_file('resources/pts2d-pic_a.txt')
    pts3d = load_file("resources/pts3d.txt")

    M_8, res_8 = best_M(pts2d, pts3d, calibpts=8, testpts=4, iter=10)
    M_12, res_12 = best_M(pts2d, pts3d, calibpts=12,testpts=4, iter=10)
    M_16, res_16 = best_M(pts2d, pts3d, calibpts=16, testpts=4, iter=10)
    results = OrderedDict([('res_8', res_8), ('M_8', M_8.flatten()),
                            ('res_12', res_12), ('M_12', M_12.flatten()),
                            ('res_16', res_16), ('M_16', M_16.flatten())])
    residuals = (res_8, res_12, res_16)
    Ms = (M_8, M_12, M_16)
    res, M = min((res, M) for (res, M) in zip(residuals, Ms))
    print('Residuals:\nfor 8 pts: %.5f\nfor 12 pts: %.5f\nfor 16 pts: %.5f\n'%(res_8, res_12, res_16))
    print('Best Projection Matrix\nM (Projection Matrix) =\n%s\n'%M)
    #Estimating the camera center
    Q = M[:, :3]
    m4 = M[:, 3]
    C = np.dot(-np.linalg.inv(Q), m4)
    print('Center of Camera = %s\n'%C)
    print("Task 1 executed successfully\n")


def task_2(): #Estimating fundamental matrix
    print("Executing task: 2 \n==================")
    #reading input images
    print("Fundamental Matrix and epipolar lines using Method-1")
    img_a = cv2.imread('resources/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('resources/pic_b.jpg', cv2.IMREAD_COLOR)
    #Reading image points
    a_pts = np.array(load_file('resources/pts2d-pic_a.txt'))
    b_pts = np.array(load_file('resources/pts2d-pic_b.txt'))

    F = least_squares_F_solver(a_pts, b_pts)
    print('Fundametal Matrix with Rank = 3: \n%s\n'%F)

    F = fun_mat_rank2(F=F)
    print('Fundametal Matrix with Rank = 2: \n%s\n'%F)

    a_pts = np.column_stack((a_pts, np.ones(a_pts.shape[0])))
    b_pts = np.column_stack((b_pts, np.ones(a_pts.shape[0])))

    img_a, img_b = draw_eplines(img_a=img_a, img_b=img_b, F=F, a_pts=a_pts, b_pts=b_pts)
    cv2.imshow('Right View - M1', img_a);
    cv2.imshow('Left View - M1', img_b); 
    cv2.waitKey(0); 
    cv2.destroyAllWindows();
    cv2.imwrite('output/pic_a_M1.png',img_a)
    cv2.imwrite('output/pic_b_M1.png',img_b)


    print("Fundamental Matrix and epipolar lines using Method-2(Better fundamental matrix)")
    a_pts = np.array(load_file('resources/pts2d-pic_a.txt'))
    b_pts = np.array(load_file('resources/pts2d-pic_b.txt'))
    #Normalization Matrices
    m_a = np.mean(a_pts, axis=0)
    m_b = np.mean(b_pts, axis=0)
    a_pts_temp = np.subtract(a_pts, m_a[None,:])
    b_pts_temp = np.subtract(b_pts, m_b[None,:])
    s_a = 1 / np.abs(np.std(a_pts_temp, axis=0)).max()
    s_b = 1 / np.abs(np.std(b_pts_temp, axis=0)).max()
    S_a = np.diag([s_a, s_a, 1])
    S_b = np.diag([s_b, s_b, 1])
    C_a = np.array([[1, 0, -m_a[0]],[0, 1, -m_a[1]],[0, 0, 1]])
    C_b = np.array([[1, 0, -m_b[0]],[0, 1, -m_b[1]],[0, 0, 1]])
    T_a = np.dot(S_a, C_a)
    T_b = np.dot(S_b, C_b)

    #Homogenous Co-ordinates
    a_pts = np.column_stack((a_pts_temp, np.ones(a_pts.shape[0])))
    b_pts = np.column_stack((b_pts_temp, np.ones(b_pts.shape[0])))
    #Normzalized
    a_pts_norm = np.dot(a_pts, T_a)
    b_pts_norm = np.dot(b_pts, T_b)
    F = least_squares_F_solver(a_pts_norm, b_pts_norm)#Fundamental Matrix
    F = fun_mat_rank2(F=F)#Getting F to rank-2

    print('Ta=\n%s\n'%T_a)
    print('Tb=\n%s\n'%T_b)
    print('F=\n%s\n'%F)
    #Reading the images again
    img_a = cv2.imread('resources/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('resources/pic_b.jpg', cv2.IMREAD_COLOR)

    F = np.dot(T_b.T, np.dot(F, T_a)) #Finer fundamental matrix
    img_a, img_b = draw_eplines(img_a=img_a, img_b=img_b, F=F, a_pts=a_pts, b_pts=b_pts)

    cv2.imshow('Right View - M2', img_a);
    cv2.imshow('Left View - M2', img_b); 
    cv2.waitKey(0); 
    cv2.destroyAllWindows();
    cv2.imwrite('output/pic_a_M2.png',img_a)
    cv2.imwrite('output/pic_b_M2.png',img_b)

    print("Task 2 executed successfully")

if __name__ == '__main__':
    tasks = dict({"1":task_1,"2":task_2});
    print("Please select a task from the following")
    print("1: Estimate Camera Projection Matrix \n2: Estimate fundamental matrix and draw epipolar line \nTo execute both tasks type 0")
    sel = input("==> ")
    if sel == "0":
        tasks["1"]()
        tasks["2"]()
    else:
        try:
            print('Initializing task:',sel,'\n==================')
            tasks[sel]()
        except KeyError:
            print("INVALID TASK NAME: Please select a valid task")
