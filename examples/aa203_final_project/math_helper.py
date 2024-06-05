import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

def cross_matrix(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])

def jnp_cross_matrix(a):
    return jnp.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def jnp_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / jnp.linalg.norm(vector)

def angle_between_vectors(v1,v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def jnp_angle_between_vectors(v1,v2):
    v1_u = jnp+unit_vector(v1)
    v2_u = jnp+unit_vector(v2)
    return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

def ypr2quat(yaw, pitch, roll):
    # Create a rotation object from Euler angles specifying axes of rotation
    q_xyzw = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False).as_quat()
    return np.hstack((q_xyzw[-1],q_xyzw[:3])) #scipy puts scalar in last index, but our dynamic want scalar first
    
def quat_prod(q1,q2):
    #assumes q0 is the scalar
    return np.hstack((q1[0]*q2[0] - np.dot(q1[1:],q2[1:]),
                     q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])))

def jnp_quat_prod(q1,q2):
    #assumes q0 is the scalar
    return jnp.hstack((q1[0]*q2[0] - jnp.dot(q1[1:],q2[1:]),
                     q1[0]*q2[1:] + q2[0]*q1[1:] + jnp.cross(q1[1:], q2[1:])))

def quat_conj(quat):
    #assumes q0 is the scalar
    return np.hstack((quat[0], -quat[1:]))

def jnp_quat_conj(quat):
    #assumes q0 is the scalar
    return jnp.hstack((quat[0], -quat[1:]))

def rotate_btof(x_in_b, quat):
    return quat_prod(quat_prod(quat, np.hstack((0, x_in_b))), quat_conj(quat))[1:]

def jnp_rotate_btof(x_in_b, quat):
    return jnp_quat_prod(jnp_quat_prod(quat, jnp.hstack((0, x_in_b))), jnp_quat_conj(quat))[1:]

def quat_dot(quat, w_in_b):
    G = np.array([[-quat[1],  quat[0],  quat[3], -quat[2]],
                   [-quat[2], -quat[3],  quat[0],  quat[1]],
                   [-quat[3],  quat[2], -quat[1],  quat[0]]])
    return 1/2*np.transpose(G) @ w_in_b

def jnp_quat_dot(quat, w_in_b):
    G = jnp.array([[-quat[1],  quat[0],  quat[3], -quat[2]],
                   [-quat[2], -quat[3],  quat[0],  quat[1]],
                   [-quat[3],  quat[2], -quat[1],  quat[0]]])
    return 1/2*jnp.transpose(G) @ w_in_b

def calc_J(mass, a, b, c):
    J = mass/5 * np.array([[b**2+c**2,         0,         0],
                           [        0, a**2+c**2,         0],
                           [        0,         0, a**2+b**2]])
    Jinv = 5/mass * np.array([[1/(b**2+c**2),             0,             0],
                              [            0, 1/(a**2+c**2),             0],
                              [            0,             0, 1/(a**2+b**2)]])
    return J, Jinv

def jnp_calc_J(mass, a, b, c):
    J = mass/5 * jnp.array([[b**2+c**2,         0,         0],
                            [        0, a**2+c**2,         0],
                            [        0,         0, a**2+b**2]])
    Jinv = 5/mass * jnp.array([[1/(b**2+c**2),             0,             0],
                               [            0, 1/(a**2+c**2),             0],
                               [            0,             0, 1/(a**2+b**2)]])
    return J, Jinv