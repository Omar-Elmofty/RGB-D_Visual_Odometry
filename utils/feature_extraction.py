""" File containing methods for extracting and matchin orb features 
from camera frames

2020-01-20 -- Omar Elmofty: Created file
"""

import numpy as np


def generate_assoc_list(path):
    """Function for generating associate list containing pairs of 
    matched rgb image and depth map names based on the time they were
    captured

    Args:
        path: path to the dataset

    Returns:
        assoc_list: the list of matched rgb images and depth map names
    """

    path = path + '/rgbd_assoc.txt'
    f = open(path,'r')

    assoc_list = f.readlines()

    return assoc_list


def extract_features(assoc_list,i,n_skip,num_features=1000 ,z_thres=5,direct=''):
    """Function that extracts orb features from consecutive camera 
    frames, matches them using brute force matcher, then extracts the 
    depth of the matched keypoints from Microsoft kinect depth map

    Args:
        assoc_list: the list of matched rgb images and depth map names
                    at a specific time step
        i: index of frame k-1 in associate list
        n_skip: number of frames to skip, frame k will have an index 
                i+n_skip
        num_features: number of orb features to capture in each frame
        z_thres: the max value of depth to include
        direct: directory of dataset

    Returns:
        kps: a dictionary of matched keypoints between camera frames
        im1_name: name of first image in the image pair
    """
    
    kernel = np.array([[-2,-2,-2], 
                   [-2, 18,-2],
                   [-2,-2,-2]])
    
    im1_name = assoc_list[i].split()[1]
    dp1_name = assoc_list[i].split()[3]
    dp2_name = assoc_list[i+n_skip].split()[3] 
    im2_name = assoc_list[i+n_skip].split()[1]
    
    #Read RGB and depth images
    img1 = cv2.imread(direct.format(im1_name))#,cv2.IMREAD_GRAYSCALE)
    #img1 = cv2.filter2D(img1, -1, kernel)
    img2 = cv2.imread(direct.format(im2_name))#, cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.filter2D(img2, -1, kernel)

    dp1 = cv2.imread(direct.format(dp1_name),cv2.IMREAD_ANYDEPTH)
    dp2 = cv2.imread(direct.format(dp2_name), cv2.IMREAD_ANYDEPTH)
    
    # ORB Detector
    orb = cv2.ORB_create(nfeatures=num_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    try:
        # Brute Force Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
    except:
        print('No descriptors found')
        return [],im1_name
    
    kps = []
    
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        # Get the coordinates
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        #Extract depth for each pixel
        z1 = dp1[int(y1)][int(x1)]/5000.0
        z2 = dp2[int(y2)][int(x2)]/5000.0
        # Append to each list
        if z1>0 and z2>0 and z1 <z_thres and z2<z_thres:
            kps.append({
                'yk-1':np.array([[x1],
                                 [y1],
                                 [z1],
                                 [1]]),
                'yk':np.array([[x2],
                               [y2],
                               [z2],
                               [1]])})
    return kps,im1_name


def generate_ground_truth_trajectory(path):
    """Function that reads the ground truth data and generates the
    ground truth trajectory

    Args:
        path: path to data-set
    Returns:
        t_true: an array of all the timesteps 
        x_true: an array of the x positions for the ground truth 
        y_true: an array of the y positions for the ground truth
    """

    path = path + "/groundtruth.txt"
    f = open(path)
    t_true = []
    x_true = []
    y_true = []

    i=0
    for line in f.readlines():
        if i > 2:
            line = line.split()
            t_true.append(float(line[0]))
            x_true.append(float(line[1]))
            y_true.append(float(line[2]))
        i +=1
    f.close()

    t_true = np.array(t_true)
    x_true = np.array(x_true)
    y_true = np.array(y_true)

    return t_true, x_true, y_true