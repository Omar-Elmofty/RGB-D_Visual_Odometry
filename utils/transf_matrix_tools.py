import numpy as np


def skew_sym_3(v):
	"""Function that applies the skew symmetric operator onto a 3 
	dimensional column vector

    Args:
        v: 3 dimenstional column vector

    Returns:
        3x3 matrix: the output of the skew symmetric operation

    """
    
    return np.array([[0,-v[2][0],v[1][0]],
                     [v[2][0],0,-v[0][0]],
                     [-v[1][0],v[0][0],0]])



def skew_sym_6(v):
	"""Function that applies the skew symmetric operator onto a 6 
	dimensional column vector

    Args:
        v: 6 dimenstional column vector

    Returns:
        4x4 matrix: the output of the skew symmetric operation

    """

    row = np.array([[v[0][0]],
                    [v[1][0]],
                    [v[2][0]]])
    phi = np.array([[v[3][0]],
                    [v[4][0]],
                    [v[5][0]]])
    phi_sk_sym = skew_sym_3(phi)
    
    top = np.concatenate((phi_sk_sym,row),axis=1)
    bott = np.array([[0,0,0,1]])
    M = np.concatenate((top,bott),axis=0)
    
    return M


def calc_C(phi):
	"""Function that calculates rotation matrix C from the lie algebra
	phi

    Args:
        phi: 3 dimenstional column vector (lie algebra)

    Returns:
        C: 3x3 rotation matrix

    """

    norm = np.linalg.norm(phi)
    a = phi/float(norm)
    v_one = np.zeros((3,3))
    v_one[0][0]=1
    v_one[1][1]=1
    v_one[2][2]=1
    
    a_sk_sym = skew_sym_3(a)
    aaT= np.dot(a,np.transpose(a))
    
    C = np.cos(norm)*v_one +(1-np.cos(norm))*aaT+np.sin(norm)*a_sk_sym
    
    return C

def calc_J(phi):
	"""Function that calculates J matrix from the lie algebra phi,
	the J matrix is used for getting the translation vector between 
	reference frames

    Args:
        phi: 3 dimenstional column vector (lie algebra)

    Returns:
        J: 3x3 matrix

    """

    norm = np.linalg.norm(phi) 
    a = phi/float(norm)
    v_one = np.identity(3)
   
    
    a_sk_sym = skew_sym_3(a)
    aaT= np.dot(a,np.transpose(a))
    
    J = np.sin(norm)/float(norm)*v_one+(1-np.sin(norm)/float(norm))*aaT+(1-np.cos(norm))/float(norm)*a_sk_sym
    
    return J    


def calc_exp_v1tov2(v):
	"""Function that calculates the exponential mapping between the 
	6 dimensional lie algebra and the 4x4 rotation matrix. The functions maps points from frame v1 to frame v2.	

    Args:
        v: 6 dimenstional column vector (lie algebra)

    Returns:
        M: 4x4 Transformation matrix

    """

    row = np.array([[v[0][0]],
                    [v[1][0]],
                    [v[2][0]]])
    phi = np.array([[v[3][0]],
                    [v[4][0]],
                    [v[5][0]]])
    C = calc_C(phi)
    
    J = calc_J(phi)
    r = np.dot(J,row)
    r = np.dot(-1*np.transpose(C),r) #added this step to resolve the issue, based on book page 319
    
    top = np.concatenate((np.transpose(C),r),axis=1) # changed to transpose
    bott = np.array([[0,0,0,1]])
    M = np.concatenate((top,bott),axis=0)
    
    return M


def calc_exp_v2tov1(v):
	"""Function that calculates the exponential mapping between the 
	6 dimensional lie algebra and the 4x4 rotation matrix. The functions maps points from frame v2 to frame v1.	

    Args:
        v: 6 dimenstional column vector (lie algebra)

    Returns:
        M: 4x4 Transformation matrix

    """

    row = np.array([[v[0][0]],
                    [v[1][0]],
                    [v[2][0]]])
    phi = np.array([[v[3][0]],
                    [v[4][0]],
                    [v[5][0]]])
    C = calc_C(phi)
    #sanity check for C:
    #test = np.dot(C,np.transpose(C))
    #print('results',test)
    J = calc_J(phi)
    r = np.dot(J,row)
    
    top = np.concatenate((C,r),axis=1) # changed to transpose
    bott = np.array([[0,0,0,1]])
    M = np.concatenate((top,bott),axis=0)
    
    return M


def T_inv(T):
	"""Function that calculates the inverse of a transformation matrix

    Args:
        T: 4x4 transformation matrix

    Returns:
        T_new: inverse of T, also 4x4

    """

    T_new = np.zeros((4,4))
    C = T[0:3,0:3]
    r = T[0:3,3]
    
    T_new[0:3,0:3]=C.T
    T_new[0:3,3]=np.dot(-C.T,r)
    
    T_new[3,3]=1
    
    return T_new


def calc_ln_3(C):
	"""Function that calculates the inverse mapping from rotation matrix
	to lie algebra

    Args:
        C: 3x3 rotation matrix

    Returns:
        phi: rotation maginitude 
        a: rotation unit vector (3 dimensional column vector)
    """

    
    tr = np.matrix.trace(C)
    
    phi = np.arccos((tr-1)/2.0)
    
    w, v = np.linalg.eig(C)

    for i in range(3):
        a = v[:,i:i+1]
        if (a[0,0].imag == 0) and (a[1,0].imag == 0) and (a[2,0].imag == 0):
            break
    C_fwd = calc_C(phi*a)

    return phi, a


def calc_ln_6(T):
	"""Function that calculates the inverse mapping from transformation
	matrix to lie algebra

    Args:
        T: 4x4 Transformation matrix

    Returns:
        psi: (6x1) lie algebra 
    """

    C = T[0:3,0:3]
    r = T[0:3,3:4]
    phi,a = calc_ln_3(C)
    J = calc_J(phi*a)
    rho = np.dot(np.linalg.inv(J),r)
    
    psi = np.concatenate((rho,phi*a),axis=0)
    
    return psi