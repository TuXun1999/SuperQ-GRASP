
import numpy as np
import math
import scipy
from scipy.spatial.transform import Rotation as R
import pickle
import os
import torch
'''
The file containing all the necessary helper functions in processing sq's
'''
########
## Part I: Handle a sq at the initial pose (i.e. SE(3)= I)
########
def create_superellipsoids(e1, e2, a1, a2, a3):
    pc_shape = 50
    pc = np.zeros((pc_shape * pc_shape, 3))

    eta = np.linspace(-np.pi/2, np.pi/2, pc_shape, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, pc_shape, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)

    t1 = np.sign(np.cos(eta)) * np.power(abs(np.cos(eta)), e1)
    t2 = np.sign(np.sin(eta)) * np.power(abs(np.sin(eta)), e1)
    t3 = np.sign(np.cos(omega)) * np.power(abs(np.cos(omega)), e2)
    t4 = np.sign(np.sin(omega)) * np.power(abs(np.sin(omega)), e2)
    pc[:, 0] = (a1 * t1 * t3).flatten()
    pc[:, 1] = (a2 * t1 * t4).flatten()
    pc[:, 2] = (a3 * t2).flatten()

    return pc

def grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, sample_number, tolerance, reflection=0):
    '''
    The function to sample several grasp poses at the specified sampled angles
    Input: reflection: the direction to reflect the sampled grasp poses
    angle_sample_space_num: the total number of angle candidates at the quarter
    sample_number: the expected number of grasp candidates sampled in the quarter
    tolerance: how far the gripper is away from the edge of the shape
    '''
    if e >= 1: # The easy case, where no "shrinking" occurs
        angle = np.linspace(0, np.pi/2, num=angle_sample_space_num)
        angle_sample = angle[np.random.randint(0, angle_sample_space_num, sample_number)]

        # Obtain a quarter of the final grasp candidates
        # The cos & sin values are always non-negative, because we only consider 0 - pi/2
        res_x = (a1 + tolerance) * np.power(np.cos(angle_sample), e)
        res_y = (a2 + tolerance) * np.power(np.sin(angle_sample), e)

        # Convert the results into columns
        res_x = np.asarray(res_x).reshape(-1, 1)
        res_y = np.asarray(res_y).reshape(-1, 1)

        # Force the grippers to look into the origin
        angle_sample = np.arctan2(res_y, res_x).reshape(-1, 1)
        
    else:
        # TODO: figure out the situation where e<1, and there will be inconsistent parts
        # Use straight lines to fit the "empty" parts

        angle = np.linspace(0, np.pi/2, num=angle_sample_space_num)
        angle_sample = angle[np.random.randint(0, angle_sample_space_num, sample_number)]
        x_der = e * a1 * np.power(np.cos(angle), e - 1) * np.sin(angle)
        y_der = e * a2 * np.power(np.sin(angle), e - 1) * np.cos(angle)

        # Find the angle values where x_der, y_der boost up 
        angle_critical_x_idx = np.argmin(abs(x_der - 4 * e * a1)) # The critical angle where dx boosts up
        angle_critical_x = angle[angle_critical_x_idx]
        angle_critical_y_idx  = np.argmin(abs(y_der - 4 * e * a2)) # The critical angle where dy boosts up
        angle_critical_y = angle[angle_critical_y_idx]

        # The critical points
        # The critical point for the right discontinuous part
        right_uncont_x0 = a1 * np.power(np.cos(angle_critical_y), e)
        right_uncont_y0 = a2 * np.power(np.sin(angle_critical_y), e)

         # Find the critical point for the top discontinuous part
        top_uncont_x0 = a1 * np.power(np.cos(angle_critical_x), e)
        top_uncont_y0 = a2 * np.power(np.sin(angle_critical_x), e)


        # Sample grasp candidates in the continuous region at first
        right_uncont_angle = np.arctan2(right_uncont_y0, right_uncont_x0)
        top_uncont_angle = np.arctan2(top_uncont_y0, top_uncont_x0)

        # According to the ratio of the continuous region,
        # determine the number of grasp candidates in that part
        cont_ratio = (top_uncont_angle - right_uncont_angle) / (np.pi/2)
        angle_sample_cont_num = (int)(cont_ratio * sample_number)

        # Only sample a few candidates from the continuous part
        angle_sample_cont = angle[\
            np.random.randint(angle_critical_y_idx, \
                              angle_critical_x_idx, \
                                angle_sample_cont_num)]
        res_x_cont = (a1 + tolerance) * np.power(np.cos(angle_sample_cont), e)
        res_y_cont = (a2 + tolerance) * np.power(np.sin(angle_sample_cont), e)

        # Sample grasp candidates in the un-continuous regions
        uncont_cand_space = 20

        # Find the number of candidates from the right un-continuous part
        right_sample_num = (int)((sample_number - angle_sample_cont_num)/ 2)
        right_theta = np.arctan2(a1 - right_uncont_x0, right_uncont_y0)

        # The candidate space for the right uncontinuous part
        right_uncont_xcoord = np.linspace(right_uncont_x0, a1, uncont_cand_space)
        
        right_uncont_xcoord = right_uncont_xcoord[\
            np.random.randint(0, uncont_cand_space - 1, right_sample_num)]
        # Use geometric relationship to sample several candidates
        right_uncont_ycoord = (right_uncont_y0 / (a1 - right_uncont_x0)) * (a1 - right_uncont_xcoord)
        res_x_right_uncont = right_uncont_xcoord + tolerance * np.cos(right_theta)
        res_y_right_uncont = right_uncont_ycoord + tolerance * np.sin(right_theta)

        angle_sample_right_uncont = np.ones(right_sample_num) * (right_theta)
       
        top_sample_num = sample_number - angle_sample_cont_num - right_sample_num
        top_theta = np.arctan2(a2 - top_uncont_y0, top_uncont_x0)

        # The candidate space for the top uncontinuous part
        top_uncont_ycoord = np.linspace(top_uncont_y0, a2, uncont_cand_space)
        top_uncont_ycoord = top_uncont_ycoord[\
            np.random.randint(0, uncont_cand_space - 1, top_sample_num)]
        
        # Use geometric properties to determine the locations of the points
        top_uncont_xcoord = (top_uncont_x0 / (a2 - top_uncont_y0)) * (a2 - top_uncont_ycoord)
        res_y_top_uncont = top_uncont_ycoord + tolerance * np.cos(top_theta)
        res_x_top_uncont = top_uncont_xcoord + tolerance * np.sin(top_theta)
        
        angle_sample_top_uncont = np.ones(top_sample_num) * (np.pi/2 - top_theta)
        # Stack all results together
        res_x = np.hstack((res_x_right_uncont, res_x_cont, res_x_top_uncont))
        res_y = np.hstack((res_y_right_uncont, res_y_cont, res_y_top_uncont))
        angle_sample = np.hstack((angle_sample_right_uncont, \
                                  angle_sample_cont, angle_sample_top_uncont))
        
        # Reshape the results into columns
        res_x = res_x.reshape(-1, 1)
        res_y = res_y.reshape(-1, 1)
        angle_sample = angle_sample.reshape(-1, 1)
    

   

    # Do the necessary reflections
    if reflection == 0: # Quarter I
        res = np.hstack((res_x, res_y, angle_sample))
    elif reflection == 1: # Quarter II
        res = np.hstack((-res_x, res_y, np.pi - angle_sample))
    elif reflection == 2: # Quarter III
        res = np.hstack((-res_x, -res_y, np.pi + angle_sample))
    elif reflection == 3: # Quarter IV
        res = np.hstack((res_x, -res_y, 2 * np.pi - angle_sample))
    else:
        print("Unknown reflection direction")
        assert False
    assert res.shape[1] == 3
    return res
def grasp_pose_predict_sq(a1, a2, e, sample_number = 20, tolerance=0.5):
    '''
    The function to predict several grasp poses for a superquadric on xy-plane
    Only for grasp poses on the plane
    Input: a1, a2: the lengths of the two axes
            e: the power of the sinusoidal terms
    '''
    angle_sample_space_num = 50
    
    # Reflect the results to the other regions
    res_part1 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                          (int)(sample_number/4), tolerance, reflection=0)
    res_part2 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=1)
    res_part3 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=2)
    res_part4 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=3)

    res = np.vstack((res_part1, res_part2, res_part3, res_part4))
    return res

def transform_matrix_convert(grasp_poses, principal_axis):
    '''
    The function to convert the grasp poses from grasp pose prediction (sq) 
    to standard 4x4 transformation matrices
    '''
    result = []
    for i in range(grasp_poses.shape[0]):
        x, y, angle = grasp_poses[i]
        rot = R.from_quat([0, 0, np.sin(angle/2), np.cos(angle/2)]).as_matrix()
        tran = np.vstack((np.hstack((rot, np.array([x,y,0]).reshape(-1,1))), np.array([0, 0, 0, 1])))

        if principal_axis == 2: # z is the shortest axis in length
            pass
        elif principal_axis == 1: # y is the shortest axis in length
            tran = np.matmul(np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]), tran)
        elif principal_axis == 0: # x is the shortest axis in length
            tran = np.matmul(np.array([
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]), tran)
        result.append(tran)
    return result

############
## Part II: Read sq paraemters
############
def read_sq_parameters(filename):
    '''
    The function to read the superquadrics parameters from the file
    '''
    filename_suffix = filename.split('.')[-1]
    parameters = None
    if filename_suffix == "p": # If the file is stored as a pickle file
        with (open(filename, "rb")) as openfile:
            parameters = pickle.load(openfile)
            
    return parameters


def read_sq_directory(sq_dir, norm_scale, norm_d):
    '''
    The function to read a series of sq from the given dir in "sq_dir"
    Input:
    norm_scale: the scale factor used in normalization
    norm_d: the displacement vector used in normalization
    '''
    sq_vertices = []
    sq_transformation = []
    norm = norm_scale
    c = norm_d
    # Read the sq attributes in each file
    for sq_file in os.listdir(sq_dir):
        parameters = read_sq_parameters(sq_dir + "/" + sq_file)
        if parameters is None:
            continue 
        prob = parameters["probability"][0]
        if prob < 0.5:
            continue

        # Read the parameters
        epsilon1 = parameters["shape"][0]
        epsilon2 = parameters["shape"][1]
        a1 = parameters["size"][0]
        a2 = parameters["size"][1]
        a3 = parameters["size"][2]

        # Sample the points on the superquadric
        pc = create_superellipsoids(epsilon1, epsilon2, a1, a2, a3)

        # Apply the transformation
        quat = parameters["rotation"]
        quat = np.array(quat)
        quat[[0, 1, 2, 3]] = quat[[1, 2, 3, 0]]


        translation = np.asarray(parameters["location"])
        
        rot = R.from_quat(quat).as_matrix()

        # Obtain the correct point coordinates in the normalized frame
        pc_tran = rot.T.dot(pc.T) + translation.reshape(3, -1)
        pc_tran = pc_tran.T
        sq_tran = np.vstack((np.hstack((rot.T, translation.reshape(3, -1))), np.array([0, 0, 0, 1])))

        # Revert the normalization process
        pc_tran = pc_tran * norm + c
        sq_tran[0:3, 3] = sq_tran[0:3, 3] * norm + c

        sq_vertices.append(pc_tran)
        sq_transformation.append({"sq_parameters": parameters, \
                                    "points": pc_tran, \
                                    "transformation": sq_tran})
    return sq_vertices, sq_transformation

def read_sq_mp(sq_mp, norm_scale, norm_d):
    '''
    The function to read the predicted superquadrics from Marching Primitives
    Input:
    sq_mp: the array containing parameters for all sq's (Nx11)
    Format: e1, e2, a1, a2, a3, r, p, y, tx, ty, tz
    normalize_stats: the statistics for normalized => restore the normalize process
    '''
    sq_vertices = []
    sq_transformation = []
    norm = norm_scale
    c = norm_d

    def eul2rotm(eul):
        '''
        The function to convert euler angles into a rotation matrix
        '''
        Rot = np.zeros((3,3))
        ct = np.cos(eul)
        st = np.sin(eul)
        Rot[0, 0] = ct[1] * ct[0]
        Rot[0, 1] = st[2] * st[1] * ct[0] - ct[2] * st[0]
        Rot[0, 2] = ct[2] * st[1] *ct[0] + st[2] * st[0]
        Rot[1, 0] = ct[1] * st[0]
        Rot[1, 1] = st[2] * st[1] * st[0] + ct[2]*ct[0]
        Rot[1, 2] = ct[2] * st[1]* st[0] - st[2]*ct[0]
        Rot[2, 0] = -st[1]
        Rot[2, 1] = st[2] * ct[1]
        Rot[2, 2] = ct[2] * ct[1]     

        return Rot  
    # Read the sq attributes in each file
    for i in range(sq_mp.shape[0]):
        # Read the superquadrics' parameters
        parameters = {}
        e1, e2, a1, a2, a3, r, p, y, t1, t2, t3 = sq_mp[i, :]
        if e1 < 0.01:
            e1 = 0.01
        if e2 < 0.01:
            e2 = 0.01

        # Basic parameters for shape
        parameters["shape"] = np.array([e1, e2])
        parameters["size"] = np.array([a1, a2, a3])

        # Parameters for transformation
        translation = np.array([t1, t2, t3])
        parameters["location"] = translation

        rot = eul2rotm(np.array([r, p, y]))
        parameters["rotation"] = rot

        # Custom function to sample points on the superquadrics
        pc = create_superellipsoids(e1, e2, a1, a2, a3)

        # Apply the transformation

        # Obtain the correct point coordinates in the normalized frame
        pc_tran = np.matmul(rot, pc.T) + translation.reshape(3, -1)
        pc_tran = pc_tran.T
        sq_tran = np.vstack((np.hstack((rot, translation.reshape(3, -1))), np.array([0, 0, 0, 1])))

        # Revert the normalization process
        pc_tran = pc_tran * norm + c
        sq_tran[0:3, 3] = sq_tran[0:3, 3] * norm + c

        sq_vertices.append(pc_tran)
        sq_transformation.append({"sq_parameters": parameters, \
                                    "points": pc_tran, \
                                    "transformation": sq_tran})
    return sq_vertices, sq_transformation


def read_mp_parameters(filename):
    '''
    The function to read the parameters pre-stored in the file
    '''
    with (open(filename, "rb")) as openfile:
        parameters = pickle.load(openfile)
            
    return parameters["sq_vertices_original"], \
        parameters["sq_transformation"],\
        parameters["normalize_stats"]
def store_mp_parameters(filename, sq_vertices_original, sq_transformation, normalize_stats):
    '''
    The function to store the parameters from MP in the pickle file
    '''
    a = {
        "sq_vertices_original": sq_vertices_original,
        "sq_transformation": sq_transformation,
        "normalize_stats": normalize_stats
    }
    with open(filename, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############
### Part III: Handle a group of sq's in space, at arbitrary location
##############  
def find_sq_closest(point_select, sq_transformation, norm_scale = 1, displacement = 0):
    '''
    The function to find the closest sq to the selected point, as the associated sq to
    the selected point
    Input: point_select: selected point
    sq_transformation: the information on ALL sq's (the displacement observed from experiments not applied)
    norm_scale: scaling factor used in normalization
    norm_d: displacement used in normalization
    displacement: the final displacement between the predicted reconstructed object and the original model
    (Observed from experiments)
    '''
    # Assign the values to the variables
    norm = norm_scale
    pos = point_select
    dist_min = np.Infinity
    sq_closest = None
    idx = 0
    correct_idx = 0
    # Read each sq and determine the "best" one
    for sq_tran_dict in sq_transformation:
        # Read the transformation of this sq & several sample points on it
        sq_tran = sq_tran_dict["transformation"]
        pc_tran = sq_tran_dict["points"]
        parameters = sq_tran_dict["sq_parameters"]

        

        # Read the parameters of this sq
        epsilon1 = parameters["shape"][0]
        epsilon2 = parameters["shape"][1]
        a1 = parameters["size"][0] * norm
        a2 = parameters["size"][1] * norm
        a3 = parameters["size"][2] * norm
        
        # Find the point's coordinate in the sq's frame
        sq_rot = sq_tran[0:3, 0:3]
        sq_t = sq_tran[0:3, 3].reshape(3, -1)
        pos_sq = np.matmul(sq_rot.T, np.array([\
                    [pos[0]], [pos[1]], [pos[2]]]\
                )) - np.matmul(sq_rot.T, sq_t)
        
        # Find the sampled points on sq in the sq's frame
        pos_sq_points = np.matmul(sq_rot.T, pc_tran.T)\
                    - np.matmul(sq_rot.T, sq_t)
        
        # NOTE: Brute force: Directly obtain the distances and find the minimum one
        dist = np.min(np.linalg.norm(pos_sq - pos_sq_points, axis=0))

        # NOTE: Analytical approximate method
        # x,y,z,_ = pos_sq.flatten()
        
        # # Calculate the evaluation value
        # x1 = x/a1
        # y1 = y/a2
        # z1 = z/a3

        # # Calculate the evaluation value F(x0, y0, z0)
        # val1 = np.power(x1*x1, 2/epsilon2) + np.power(y1*y1, 2/epsilon2)
        # val2 = np.power(val1, epsilon2/epsilon1) + np.power(z1*z1, 2/epsilon1)
        # # beta calculation
        # beta = np.power(val2, -epsilon1/2)

        # dist = abs(1 - beta) * np.sqrt(x**2 + y**2 + z**2)

        # Apply the final displacement (observed in several experiments)
        sq_tran[0:3, 3] = sq_tran[0:3, 3] + displacement
        pc_tran = pc_tran + displacement
        # Find the closest sq iteratively
        if dist < dist_min :
            dist_min = dist
            sq_closest = {}
            sq_closest["sq_parameters"] = parameters
            sq_closest["transformation"] = sq_tran
            sq_closest["points"] = pc_tran
            correct_idx = idx
        idx = idx + 1
    # Return the closest sq (with the observed additional displacement applied)
    return sq_closest, correct_idx



def grasp_pose_predict_sq_closest(sq_closest, gripper_attr, norm_scale = 1, sample_number = 20):
    '''
    The function to predict grasp poses on the sq closest to the selected point
    Input: sq_closest: attributes of the closest sq
    gripper_attr: attributes of the gripper (so far, only the length of the gripper is used)
    norm_scale: scaling factor used in normalization (to restore sq into the correct scale)
    '''
    # Read the attributes of the gripper
    gripper_length = gripper_attr["Length"]
    
    norm = norm_scale
    # Read the parameters of the associated sq
    parameters_closest = sq_closest["sq_parameters"]
    epsilon1_closest = parameters_closest["shape"][0]
    epsilon2_closest = parameters_closest["shape"][1]
    a1_closest = parameters_closest["size"][0] * norm
    a2_closest = parameters_closest["size"][1] * norm
    a3_closest = parameters_closest["size"][2] * norm
    
    # Find the principal axis & Call the function to determine the grasp poses
    # Sample a series of grasp poses

    # Check the direction of the shortest axis & the second shortest
    axes_length = np.array([a1_closest, a2_closest, a3_closest])
    min_idx = axes_length.argsort()[0]
    min_idx_second = axes_length.argsort()[1]
    ratio = (axes_length[min_idx_second] - axes_length[min_idx]) / axes_length[min_idx]

    principal_axis = 2
    if min_idx == 2: # If z is the direction of the the shortest axis in length
        grasp_poses = grasp_pose_predict_sq(a1_closest, a2_closest, epsilon2_closest, \
                                            sample_number=sample_number, tolerance= gripper_length*0.3)
    elif min_idx == 1: # If y is the direction of the shortest axis in length
        grasp_poses = grasp_pose_predict_sq(a1_closest, a3_closest, epsilon1_closest, \
                                            sample_number=sample_number, tolerance=gripper_length*0.3)
        principal_axis = 1
    else: # If x is the direction of the shorest axis in length
        grasp_poses = grasp_pose_predict_sq(a2_closest, a3_closest, epsilon1_closest, \
                                            sample_number=sample_number, tolerance=gripper_length*0.3)
        principal_axis = 0
    # Transform the grasp poses to the correct positions in world frame
    grasp_poses = transform_matrix_convert(grasp_poses, principal_axis)

    # If the two shortest axes are close in length, we could predict more grasp poses
    if ratio < 0.25: # If the two axes are close in length (ratio is small)
        principal_axis_second = 2
        if min_idx_second == 2: # If z is the direction of the the shortest axis in length
            grasp_poses_second = grasp_pose_predict_sq(a1_closest, a2_closest, epsilon2_closest, tolerance= gripper_length/2)
        elif min_idx_second == 1: # If y is the direction of the shortest axis in length
            grasp_poses_second = grasp_pose_predict_sq(a1_closest, a3_closest, epsilon1_closest, tolerance=gripper_length/2)
            principal_axis_second = 1
        else: # If x is the direction of the shorest axis in length
            grasp_poses_second = grasp_pose_predict_sq(a2_closest, a3_closest, epsilon1_closest, tolerance=gripper_length/2)
            principal_axis_second = 0
        # Transform the grasp poses to the correct positions in world frame
        grasp_poses_second = transform_matrix_convert(grasp_poses_second, principal_axis_second)
        grasp_poses = grasp_poses + grasp_poses_second

    return grasp_poses


#####
## Attempt to use iou of 3d boxes from pytorch3d to filter out overlapping superquadrics
## However, there are two issues:
## 1. The environmental requirement (cuda 11.7) conflicts with torch (requires cuda 12.1)
## 2. The performance is not improved a lot => still needs to finetune the threshold
#####

# def nms_sq_bbox(sq_predict, iou_threshold = 0.5):
#     '''
#     The function to apply Nonmaximum Suppresion on the bounding boxes
#     around the predicted superquadrics

#     Input: sq_predict: parameters of the predicted sq's 
#     Each row of it follows the format:
#     [epsilon1, epsilon2, axis_length_1, axis_length_2, axis_length_3, roll, pitch, yawl
#     translation_1, translation_2, translation_3]
#     iou_threshold: the threshold used in NMS

#     Return: 
#     sq_filter: parameters of the filtered sq's
#     '''
#     # The indices of the rows to keep
#     keep_idx = []
#     bbox = []
#     # Read the sq attributes in each file
#     for i in range(sq_predict.shape[0]):
#         # Read the superquadrics' parameters
#         e1, e2, a1, a2, a3, r, p, y, t1, t2, t3 = sq_predict[i, :]
#         if e1 < 0.01:
#             e1 = 0.01
#         if e2 < 0.01:
#             e2 = 0.01

#         # Parameters for transformation
#         translation = np.array([t1, t2, t3])

#         def eul2rotm(eul):
#             '''
#             The function to convert euler angles into a rotation matrix
#             '''
#             Rot = np.zeros((3,3))
#             ct = np.cos(eul)
#             st = np.sin(eul)
#             Rot[0, 0] = ct[1] * ct[0]
#             Rot[0, 1] = st[2] * st[1] * ct[0] - ct[2] * st[0]
#             Rot[0, 2] = ct[2] * st[1] *ct[0] + st[2] * st[0]
#             Rot[1, 0] = ct[1] * st[0]
#             Rot[1, 1] = st[2] * st[1] * st[0] + ct[2]*ct[0]
#             Rot[1, 2] = ct[2] * st[1]* st[0] - st[2]*ct[0]
#             Rot[2, 0] = -st[1]
#             Rot[2, 1] = st[2] * ct[1]
#             Rot[2, 2] = ct[2] * ct[1]     

#             return Rot 
#         rot = eul2rotm(np.array([r, p, y]))

#         # Points on the bounding box of the sq
#         pc = np.array([
#             [-a1, -a2, -a3],
#             [a1, -a2, -a3],
#             [a1, a2, -a3],
#             [-a1, a2, -a3],
#             [-a1, -a2, a3],
#             [a1, -a2, a3],
#             [a1, a2, a3],
#             [-a1, a2, a3]
#         ])

#         # Apply the transformation

#         # Obtain the correct point coordinates in the normalized frame
#         pc_tran = np.matmul(rot, pc.T) + translation.reshape(3, -1)
#         pc_tran = pc_tran.T
#         bbox.append(pc_tran)

#     bbox = np.array(bbox)
    
#     # Convert the bbox to tensor, so we can use the library from pytorch3d
#     bbox = torch.from_numpy(bbox).float()

#     _, iou_3d = box3d_overlap(bbox, bbox)

#     # NMS to filter out redundant bboxes
#     idx_cands = list(range(iou_3d.shape[0]))
#     while idx_cands:
#         idx = idx_cands.pop(0)
#         keep_idx.append(idx)

#         for j in idx_cands:
#             iou = iou_3d[idx, j]
#             if iou > iou_threshold:
#                 idx_cands.remove(j)
    
#     # Return the filtered sq's
#     keep_idx = np.array(keep_idx)
#     iou_3d = iou_3d[keep_idx, :]
#     iou_3d = iou_3d[:, keep_idx]
#     return sq_predict[keep_idx, :], bbox[keep_idx, :], iou_3d


