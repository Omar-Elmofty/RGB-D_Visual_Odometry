""" An implementation of the RANSAC algorithm for performing outlier
rejection 

2020-01-20 -- Omar Elmofty: Created file
"""

import numpy as np
import random

from point_cloud_tools import generate_p_lists, point_cloud_alignment

def RANSAC(kps,n_p=3,n_iter=50,thres=0.05):
	"""Function that performs RANSAC outlier rejection for a set of 
	matched orb features between a pair of consecutive frames. The 
	algorithm uses scalar weighted point cloud alignment as the model
	by which outliers are rejected

    Args:
    	kps: a dictionary containing pairs of matched orb features 
    		between consecutive camera frames
    	n_p: number of points used to perform scalar weighted point
    		cloud alignment
    	n_iter: number of RANSAC iterations
    	thres: threshold for classifying as an inlier vs outlier
        
    Returns:
    	yk_list_opt: a list of all the inlier orb features detected in
    				frame k
    	yk_1_list_opt: a list of all the inlier orb features detected in
    				frame k-1
        pa_list_opt: a list of inlier points in euclidean space forming
        			point cloud a
        pb_list_opt: a list of inlier points in euclidean space forming
        			point cloud b
    """
    
    len_min_var = 0
    
    pa_list_opt = []
    pb_list_opt = []
    yk_list_opt = []
    yk_1_list_opt = []
    
    pa_list_all,pb_list_all = generate_p_lists(kps)
    
    for it in range(n_iter):
        indx_array =  np.random.choice(len(kps), n_p,replace=False)

        pa_list = []
        pb_list = []


        for indx in indx_array:
            pa_list.append(pa_list_all[indx])
            pb_list.append(pb_list_all[indx])

        T_final = point_cloud_alignment(pa_list,pb_list) 

        pa_list_thres = []
        pb_list_thres = []
        yk_list_thres = []
        yk_1_list_thres = []
        for i in range(len(pa_list_all)):
            pa = pa_list_all[i]
            pb = pb_list_all[i]
            pa_est = np.dot(T_final,pb)
            error = np.linalg.norm(pa_est-pa)
            if error < thres:
                pa_list_thres.append(pa)
                pb_list_thres.append(pb)
                yk_list_thres.append(kps[i]['yk'])
                yk_1_list_thres.append(kps[i]['yk-1'])
        #Compute variance of thresholded points:
        var = np.var(pa_list_thres,axis=0)
        try:
            var = np.sqrt(var[0,0]**2+var[1,0]**2)
        except:
            var=0
        
        if (len(pa_list_thres)*var >len_min_var):
            len_min_var = len(pa_list_thres)*var
            pa_list_opt = pa_list_thres
            pb_list_opt = pb_list_thres
            yk_list_opt = yk_list_thres
            yk_1_list_opt = yk_1_list_thres
            
    return yk_list_opt,yk_1_list_opt,pa_list_opt, pb_list_opt