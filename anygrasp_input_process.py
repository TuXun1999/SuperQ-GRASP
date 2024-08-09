import os, sys
import argparse
from argparse import Namespace
import copy
pyngp_path = os.getcwd() + "/instant-ngp/build"
sys.path.append(pyngp_path)
import pyngp as ngp

import numpy as np
import shutil
import argparse
import json
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from pose_estimation.pose_estimation import estimate_camera_pose
from utils.mesh_process import coordinate_correction, depth_map_mesh

from grasp_pose_prediction.Marching_Primitives.mesh2sdf_convert import mesh2sdf_csv
from grasp_pose_prediction.Marching_Primitives.MPS import add_mp_parameters
from grasp_pose_prediction.grasp_sq_mp import predict_grasp_pose_sq
from grasp_pose_prediction.grasp_contact_graspnet import \
    predict_grasp_pose_contact_graspnet, grasp_pose_eval_gripper_cg
from preprocess import instant_NGP_screenshot
sys.path.append(os.getcwd() + "/contact_graspnet_pytorch")
from PIL import Image
import open3d as o3d

if __name__ == "__main__":
    """Command line interface."""
    parser = argparse.ArgumentParser()
    # NeRF model
    parser.add_argument(
        "nerf_model",
        help="The directory containing the NeRF model to use"
    )

    parser.add_argument('--nerf-scale', \
                        type=float, default=1, \
                        help="The value used to scale the real scene into the\
                            scene contained in an unit cube used by instant-NGP")
    # Arguments for pictures
    parser.add_argument('--distance', \
                        type=float, default=1.5, \
                        help="Radius of the semi-sphere")
    options = parser.parse_args(sys.argv[1:])


    nerf_dataset = options.nerf_model

    # nerf_dataset = "./data/chair7_pm"
    # sdf_normalize_stats, csv_filename = preprocess(camera_intrinsics_dict, options.distance, nerf_dataset, \
    #            "base_upper.ingp", options)

    csv_filename = os.path.join(nerf_dataset, "target_obj.csv")

    anygrasp_input = "./anygrasp_input"
    if os.path.exists(anygrasp_input) == True:
        shutil.rmtree(anygrasp_input)
    os.makedirs(anygrasp_input)
    nerf_model_name = nerf_dataset.split("/")[-1]
    nerf_model_name = os.path.join(anygrasp_input, nerf_model_name)
    if os.path.exists(nerf_model_name) == True:
        shutil.rmtree(nerf_model_name)
    os.makedirs(nerf_model_name)
    ######
    ## Part I: Set up the parameters
    ######
    # Obtain the mesh & Other parameters
    mesh_filename = nerf_dataset + "/target_obj.obj"
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    img_dir = nerf_dataset
    nerf_scale = options.nerf_scale


    # Select eight poses randomly from the candidates
    grasp_cands_num = 8
    h = 5
    i = 24
    grasp_cands_idx = np.random.randint((h-1)*i, size=(int)(grasp_cands_num/2))
    grasp_cands_idx2 = np.random.randint((h-1)*i, size=(int)(grasp_cands_num/2))

    # Initial transformation
    trans_initial = np.eye(4)
    trans_initial[:3, :3] = np.array(\
        [[0, 0, -1],
         [1, 0, 0],
         [0, -1, 0]])
    trans_initial[0:3, 3] = np.array([options.distance, 0, 0])

    trans_initial2 = copy.deepcopy(trans_initial)
    trans_initial2[0:3, 3] *= 1.5

    # Select the grasp poses on the semi-sphere (camera poses)
    grasp_pose_cands = []
    # Inner semi-sphere
    for idx in range(grasp_cands_idx.shape[0]):
        grasp_pose_idx = grasp_cands_idx[idx]
        angle_h = (int)(grasp_pose_idx/24) * (np.pi/(2*h))
        angle_i = (grasp_pose_idx%24) * (np.pi*2/i)
        # Horizontal Rotation
        rot1 = R.from_quat([0, np.sin(-angle_h/2), 0, np.cos(-angle_h/2)]).as_matrix()
        rot1 = np.vstack((np.hstack((rot1, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
        # Vertical Rotation
        rot2 = R.from_quat([0, 0, np.sin(angle_i/2), np.cos(angle_i/2)]).as_matrix()
        rot2 = np.vstack((np.hstack((rot2, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))

        grasp_pose_camera = rot2@(rot1@trans_initial)
        grasp_pose_cands.append(grasp_pose_camera)
    # Outer semi-sphere
    for idx in range(grasp_cands_idx2.shape[0]):
        grasp_pose_idx = grasp_cands_idx2[idx]
        angle_h = (int)(grasp_pose_idx/24) * (np.pi/(2*h))
        angle_i = (grasp_pose_idx%24) * (np.pi*2/i)
        # Horizontal Rotation
        rot1 = R.from_quat([0, np.sin(-angle_h/2), 0, np.cos(-angle_h/2)]).as_matrix()
        rot1 = np.vstack((np.hstack((rot1, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
        # Vertical Rotation
        rot2 = R.from_quat([0, 0, np.sin(angle_i/2), np.cos(angle_i/2)]).as_matrix()
        rot2 = np.vstack((np.hstack((rot2, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))

        grasp_pose_camera = rot2@(rot1@trans_initial2)
        grasp_pose_cands.append(grasp_pose_camera)
    #####
    ## Part II: Generate depth images and color images
    #####
    idx = 0
    for camera_pose_est in grasp_pose_cands[:1]:
        # Correct the frame conventions in camera & gripper of SPOT
        gripper_pose_current = camera_pose_est@np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        
        # Assume that the gripper & camera are at the same pose
        
       
        # NOTE: if  you decide to use this method, please specify 
        # the nerf snapshot (".ingp" file) and camera intrinsics
        # (In our method, they are only needed in preprocessing step)
        f = open(os.path.join(nerf_dataset, "base_cam.json"))
        camera_intrinsics_dict = json.load(f)
        f.close()
        # Obtain the camera frame (same as the gripper frame)
        camera_extrinsics = gripper_pose_current@np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        # Read the camera attributes
        fl_x = camera_intrinsics_dict["fl_x"]
        fl_y = camera_intrinsics_dict["fl_y"]
        cx = camera_intrinsics_dict["cx"]
        cy = camera_intrinsics_dict["cy"]

        # TODO: incorporate other distortion coefficients into concern as well
        camera_intrinsics_matrix = np.array([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1]
        ])
        mesh_points = mesh.vertices

        

        # Convert the camera pose to instant-NGP's convention
        camera_pose_ingp = camera_extrinsics@\
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0 , 1]])
        
        # Obtain the RGB image
        rgb_frame = instant_NGP_screenshot(nerf_dataset, "base.ingp", camera_intrinsics_dict,\
                           camera_pose_ingp, mode = "RGB")
        # Obtain the depth array in instant-NGP's scale
        # depth_array = depth_map_mesh(mesh, camera_intrinsics_matrix, camera_extrinsics)
        # # Convert the value into the real-world scene
        # depth_array /= nerf_scale
        # camera_extrinsics[0:3, 3] /= nerf_scale
        depth_frame = instant_NGP_screenshot(nerf_dataset, "base.ingp", camera_intrinsics_dict,\
                            camera_pose_ingp, mode = "Depth")
        depth_frame /= nerf_scale


        # Save them under the target folder
        image_save_path = os.path.join(nerf_model_name, "pose" + str(idx))
        if os.path.exists(image_save_path):
            shutil.rmtree(image_save_path)
        os.mkdir(image_save_path)

        rgb_save_path = os.path.join(image_save_path, "color.png")
        depth_save_path = os.path.join(image_save_path, "depth.png")

        plt.imsave(rgb_save_path, rgb_frame.copy(order='C'))
        plt.imsave(depth_save_path, depth_frame)

        idx += 1
