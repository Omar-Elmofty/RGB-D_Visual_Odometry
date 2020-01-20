""" File contains several functions used for implementing the point 
cloud alignment algorithm 

2020-01-20 -- Omar Elmofty: Created file
"""


import numpy as np

from camera_model import get_pfromy, transf_camtov

def generate_p_lists(kps):
	"""Function that generates 2 euclidean point cloud lists from pairs
	of matched orb features between 2 camera frames

    Args:
    	kps: a dictionary containing pairs of matched orb features 
    		between consecutive camera frames
        
    Returns:
        pa_list: a list of points in euclidean space forming point 
        		cloud a
        pb_list:a list of points in euclidean space forming point 
        		cloud b

    """
    
    pa_list = []
    pb_list = []
    
    for i in range (len(kps)):
        yk =kps[i]['yk']
        yk_1 = kps[i]['yk-1']
        
        #Get p from yk_1:
        p_cam = get_pfromy(yk_1)
        p_v = transf_camtov(p_cam) 
        
        pa_list.append(p_v)
        
        #Get p from yk:
        p_cam = get_pfromy(yk)
        p_v = transf_camtov(p_cam) 
        
        pb_list.append(p_v)
    return pa_list,pb_list


def generate_weights(pa_list,pb_list):
	"""Function that produces the weights for each point used for the 
	scalar weighted point cloud algorithm. Weights are inversely 
	proportional with the depth of the point

    Args:
        pa_list: a list of points in euclidean space forming point 
        		cloud a
        pb_list:a list of points in euclidean space forming point 
        		cloud b
    Returns:
        w: array containing all the weights

    """
    w = []
    for j in range(len(pa_list)):
        pa = pa_list[j]
        pb = pb_list[j]
        w.append(2/(float(0.1+pa[0,0]+pb[0,0])))
    return np.array(w)


def centroid(p_list,w_array):
	"""Function that computes the centroid of a point cloud in 
	euclidean space

    Args:
        p_list: a list of points in euclidean space forming point cloud
    	w_array: array containing all the weights

    Returns:
        avg_p: centroid (4D vector in homogeneous form [x,y,z,1])
    """
    
    sum_p = np.zeros((4,1))
    w_sum = np.sum(w_array)
    
    j = 0
    for p in p_list:
        sum_p +=w_array[j]*p
        j+=1
    avg_p = 1/float(w_sum)*sum_p
    
    return avg_p


def calc_W(pa_list,pb_list,pa_c,pb_c,w_array):
	"""Function that computes the W matrix used for the scalar weighted
	point cloud alignment algorithm.

    Args:
        pa_list: a list of points in euclidean space forming point 
        		cloud a
        pb_list:a list of points in euclidean space forming point 
        		cloud b
        pa_c: centroid of point cloud a (4x1 np.array)
        pa_b: centroid of point cloud b (4x1 np.array)
		w_array: np.array containing all the weights

    Returns:
        W: matrix W used for performing point cloud alignment
    """
    
    W = np.zeros((4,4))
    w_sum = np.sum(w_array)
    for j in range(len(pa_list)):
        pa = pa_list[j]
        pb = pb_list[j]
        W += w_array[j]*np.dot((pb-pb_c),np.transpose(pa-pa_c))
    
        
    W = 1/float(w_sum)*W
    
    W = W[0:3,0:3]
    return W


def point_cloud_alignment(pa_list,pb_list):
	"""Function that implements the scalar weighted point cloud 
	alignment algorithm,	it outputs the transformation matrix that 
	will transforms one point cloud to best fit another point cloud 
	(based on least squared errors)

    Args:
        pa_list: a list of points in euclidean space forming point 
        		cloud a
        pb_list:a list of points in euclidean space forming point 
        		cloud b
    Returns:
        T_final: The transformation matrix between point cloud a and b

    """

    w_array = generate_weights(pa_list,pb_list)
    
    pa_c = centroid(pa_list,w_array)
    pb_c = centroid(pb_list,w_array)
   
      
    W = calc_W(pa_list,pb_list, pa_c, pb_c,w_array)
    U, s, vh = scipy.linalg.svd(W, full_matrices=True)
    V = vh.T
    
    det_u = np.linalg.det(U)
    det_v = np.linalg.det(V)

    
    K = np.diag([1,1,det_u*det_v])
    
    C = np.dot(K,U.T)
    C = np.dot(V,C)
    
    C_paded= np.zeros((4,4))
    C_paded[0:3,0:3] = C
        
    r = pa_c-np.dot(C_paded, pb_c)
    T_final = np.zeros((4,4))
    T_final[0:3,0:3] = C
    T_final[0:4,3:4] = r
    
    return T_final