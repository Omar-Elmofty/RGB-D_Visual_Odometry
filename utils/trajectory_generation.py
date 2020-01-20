""" File containing several tools used for performing trajectory 
generation from incremental poses changes between camera frames

2020-01-20 -- Omar Elmofty: Created file
"""

import numpy as np


def generate_trajectory(T_list,theta=0.0001):
	"""Function that accumulates pose changes from consecutive time
	steps and generate the final trajectory.

    Args:
    	T_list: list containing pose changes at each time step, each
    			pose change is 4x4 tranformation matrix
    	theta: a paramter for rotating the generated trajectory in the
    			x-y plane 

    Returns:
        x: list of x positions at time step index k
        y: list of y positions at time step index k
    """
    
    T = calc_exp_v2tov1(np.array([0,0,0,0,0,theta]).reshape(-1,1))
    T = T[0:3,0:3]
    
    y = [0]
    x = [0]
    #T_list_new = [np.identity(4)]
    r_list= [T_list[0][0:3,3].reshape(-1,1)]
    C_list = [T_list[0][0:3,0:3]]
    for i in range(len(T_list)-1):
        C1 = C_list[i]
        C2 = T_list[i+1][0:3,0:3]
        r2 = T_list[i+1][0:3,3].reshape(-1,1)
        #T_new = T_list_new[i].dot(T_list[i+1])
        #T_list_new.append(T_new.dot(T_rot))
        r_new = np.dot(C1.T,r2)+r_list[i]
        C_new = np.dot(C2,C1)
        C_list.append(C_new)
        r_list.append(r_new)
        r_new = np.dot(T,r_new)
        x.append(r_new[0,0])
        y.append(r_new[1,0])
    return x,y


def generate_trajectory_psuedo(T_list,theta=0.0001):
	"""Function that accumulates pose changes from consecutive time
	steps and generate the final trajectory. This function is for 
	simulated measurments only and is used for testing

    Args:
    	T_list: list containing pose changes at each time step, each
    			pose change is 4x4 tranformation matrix
    	theta: a paramter for rotating the generated trajectory in the
    			x-y plane 

    Returns:
        x: list of x positions at time step index k
        y: list of y positions at time step index k
        T_list_new: list of transformation matrices from inital position
        		to positon in time k
    """
    
    y = [0]
    x = [0]
    T_list_new = [np.identity(4)]
    
    for i in range(len(T_list)-1):
      
        T_new = T_list_new[i].dot(T_list[i+1])
        T_list_new.append(T_new)
       
        x.append(T_list_new[-1][0,3])
        y.append(T_list_new[-1][1,3])
    return x,y,T_list_new

