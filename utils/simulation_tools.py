""" File contains several functiions for creating a simulated 
enviroment for testing the algorithm and estimating covariance

2020-01-20 -- Omar Elmofty: Created file
"""

import numpy as np
import random

from transf_matrix_tools import calc_exp_v2tov1


def generate_psuedo_T_List(n_p,dxrange = [0,0.01], dthrange = [-0.1,0.1]):
	"""Function that generates a simulated trajectory

    Args:
        n_p: number of time steps
        dxrange: the range for sampling delta x
        dthrange: the range for sampling thetas for generating trajectory

    Returns:
       	T_list: the list of poses forming the simulated trajectory

    """
    T_list = []
    dx = np.random.uniform(dxrange[0],dxrange[1],n_p)
    dtheta = np.random.uniform(dthrange[0],dthrange[1],n_p)
    for i in range(n_p):
        v = np.array([dx[i],0,0,0,0,dtheta[i]]).reshape(-1,1)
        T = calc_exp_v2tov1(v)
        T_list.append(T)
    return T_list


def generate_psuedo_vod_obs(T_list,n_p,sigma_pix,sigma_z):
	"""Function that generates simulated observations for the generated
	simulated trajectory

    Args:
        T_list: lists of poses forming the simulated trajectory
        n_p: number of features detected in each frame
        sigma_pix: the standard deviation used for adding noise to u,y
        			pixel measurements
        sigma_z: the standard deviation used for adding noise to the 
        		depth measurements

    Returns:
       	vod_obs_psuedo: a dictionary containing all the simulated 
       					measurements

    """
    
    vod_obs_psuedo = []
    
    for i in range (len(T_list)):
        x = np.random.uniform(0,640,n_p) 
        y = np.random.uniform(0,480,n_p) 
        z = np.random.uniform(0.5,5,n_p) 
        vod_obs_psuedo.append({'kps':[]})
        for j in range(n_p):
            yk = np.array([x[j],y[j],z[j],1]).reshape(-1,1)
            pk_cam = get_pfromy(yk)
            pk_v = transf_camtov(pk_cam)
            pk_1_v = np.dot(T_list[i],pk_v)
            pk_1_cam = transf_vtocam(pk_1_v)
            yk_1 = get_yfromp(pk_1_cam)
            #add noise to yk_1
            yk_1[0][0] += np.random.normal(0,sigma_pix)
            yk_1[1][0] += np.random.normal(0,sigma_pix)
            yk_1[2][0] += np.random.normal(0,sigma_z)
            vod_obs_psuedo[i]['kps'].append({'yk':yk,'yk-1':yk_1})
            
    return vod_obs_psuedo
