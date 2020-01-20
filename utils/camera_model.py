""" File contains several functions defining the Microsoft kinect camera 
model.

2020-01-20 -- Omar Elmofty: Created file
"""

import numpy as np


def get_yfromp(pr):
	"""Function that transforms a point coordinates from cartesean 
	space to camera measurements

    Args:
        pr: 4 dimenstional column vector of 3D position in homogenous
        form (x,y,z,1)

    Returns:
        y: camera measurements form this position (u,v,z,1)

    """

    zr = np.array([1/float(pr[2][0]),1/float(pr[2][0]),1,1])
    Zr = np.diag(zr)
    
    Pr = np.array([[520.9,0,0,325.1],
                  [0,521,0,249.7],
                  [0,0,1,0],
                  [0,0,0,1]])
    r = np.dot(Zr,pr)
    y=np.dot(Pr,r)
    return y


def get_pfromy(y):
	"""Function that transforms a camera measurements to cartesean 
	coordinates 

    Args:
    	y: camera measurements form this position (u,v,z,1)

    Returns:
    	p: 4 dimenstional column vector of 3D position in homogenous
        form (x,y,z,1)
    """

    z = np.array([float(y[2][0]),float(y[2][0]),1,1])
    Z_inv = np.diag(z)
    
    P_inv = np.array([[1/520.9,0,0,-325.1/520.9],
                  [0,1/521.0,0,-249.7/521.0],
                  [0,0,1,0],
                  [0,0,0,1]])
    r = np.dot(P_inv,y)
    p=np.dot(Z_inv,r)
    return p


def transf_camtov(p):
	"""Function that transforms points expressed in the camera frame to 
	vehicle frame

    Args:
    	p: 4 dimenstional column vector of 3D position in homogenous
        form in the camera frame (x,y,z,1)

    Returns:
    	p: 4 dimenstional column vector of 3D position in homogenous
        form in the vehicle frame (x,y,z,1)
    """

    T = np.array([[0,0,1,0],
                  [-1,0,0,0],
                  [0,-1,0,0],
                  [0,0,0,1]])
    p_new = np.dot(T,p)
    return p_new


def transf_vtocam(p):
	"""Function that transforms points expressed in the vehicle frame to 
	camera frame

    Args:
    	p: 4 dimenstional column vector of 3D position in homogenous
        form in the vehicle frame (x,y,z,1)

    Returns:
    	p: 4 dimenstional column vector of 3D position in homogenous
        form in the camera frame (x,y,z,1)
    """

    T = np.array([[0,-1,0,0],
                  [0,0,-1,0],
                  [1,0,0,0],
                  [0,0,0,1]])
    p_new = np.dot(T,p)
    return p_new