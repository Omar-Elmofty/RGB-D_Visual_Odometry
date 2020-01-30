""" Implementation of RGBD VO pipeline

2020-01-21 -- Omar Elmofty: Created file
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from feature_extraction import extract_features

if __name__ == "__main__":
	"""Implementation of the RGB visual odometry pipeline
   """

	T_list = [] #list for saving poses
	# Create video for visualization
	out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,480))

	#Incrementors for iterations
	n_skip = 1
	k = 0

	direct = 'rgbd_dataset_freiburg2_pioneer_slam3/{}'
	t_list = []
	while (k+n_skip) < len(assoc_list):
	    
	    #Ensure n_skip doesn't explode
	    if n_skip>10:
	        print('Checkpoint5: n_skip too large')
	        k+=1
	        n_skip=1
	        continue
	    
	    
	    kps,im1_name = extract_features(assoc_list,k,n_skip,num_features=5000,z_thres=5,direct=direct)
	    #Increment n_skip if no Kps detected
	    if len(kps)==0:
	        print('Checkpoint 1: No Keypoints Found')
	        n_skip +=1
	        continue         
	    
	    
	    if len(kps)>10:
	        yk_list,yk_1_list,pa_list,pb_list=RANSAC(kps,3,n_iter=40,thres=0.08)
	        n_p = len(yk_list)
	    else:
	        print('Checkpoint2: Not enough measurements, k=',k)
	        n_skip+=1
	        continue
	    
	    if len(pa_list)>10:
	        T_final = point_cloud_alignment(pa_list,pb_list)
	    else:
	        T_final = np.diag([1,1,1,1])
	        print('Checkpoint3: Not enough points, k=',k)
	        n_skip+=1
	        continue
	        
	    if np.linalg.norm(T_final[0:3,3:4])>0.2:
	        T_final = np.diag([1,1,1,1])#T_list[-1]
	        print('Checkpoint4: T_final too large,k = ',k)
	        n_skip +=1
	        continue
	    
	    
	    #Detected - append to T_list, increment k and re-set n_skip
	    T_list.append(T_final)
	    t_list.append(assoc_list[k].split()[0])
	    k+=n_skip
	    n_skip=1
	    
	    ############################################################################
	    
	    #Print Status 
	    if k%10 == 0:
	        print('Iter=',k)
	        
	    #visualize Results
	    img2 = cv2.imread(direct.format(im1_name))
	    for j in range(min(n_p,len(yk_1_list))):
	        yk_1_m = yk_1_list[j]
	        yk_m = yk_list[j]

	        p_cam = get_pfromy(yk_m)
	        p_v = transf_camtov(p_cam)
	        p_v_new = np.dot(T_final,p_v)
	        p_cam = transf_vtocam(p_v_new)
	        yk_est = get_yfromp(p_cam)

	        p_cam = get_pfromy(yk_1_m)
	        p_v = transf_camtov(p_cam)      

	        img2 = cv2.circle(img2, (int(yk_m[0][0]),int(yk_m[1][0])), 2, (0,255,0), 7)
	        img2 = cv2.circle(img2, (int(yk_1_m[0][0]),int(yk_1_m[1][0])), 2, (255,0,0), 7)
	        img2 = cv2.circle(img2, (int(yk_est[0][0]),int(yk_est[1][0])), 2, (0,0,255), 3)
	        img2 = cv2.putText(img2,'k='+str(k),org=(50, 50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0, 255))

	    #Save Videos
	    out.write(img2)
	        
	out.release()       
	print('Finished')