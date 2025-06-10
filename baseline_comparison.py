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
    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true', help="Whether to activate the visualization platform"
    )

    # Arguments for pictures
    parser.add_argument('--distance', \
                        type=float, default=4.0, \
                        help="Radius of the semi-sphere (in nerf scale)")
    parser.add_argument('--nerf-scale', \
                        type=float, default=1, \
                        help="The value used to scale the real scene into the\
                            scene contained in an unit cube used by instant-NGP")
    parser.add_argument('--iterations-per-method', type=int, default=2, \
                        help="The grasp candiates generated from  each method. 2 at minimum")
    options = parser.parse_args(sys.argv[1:])


    nerf_dataset = options.nerf_model

    # nerf_dataset = "./data/chair7_pm"
    # sdf_normalize_stats, csv_filename = preprocess(camera_intrinsics_dict, options.distance, nerf_dataset, \
    #            "base_upper.ingp", options)

    csv_filename = os.path.join(nerf_dataset, "target_obj.csv")
    ######
    ## Part I: Set up the parameters
    ######
    # Obtain the mesh & Other parameters
    mesh_filename = nerf_dataset + "/target_obj.obj"
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    img_dir = nerf_dataset
    nerf_scale = options.nerf_scale


    # Select eight poses randomly from the candidates
    grasp_cands_num = (int)(options.iterations_per_method)
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

    # The candidates for the gripper poses (camera poses)
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
    ## Part II: Embed grasp pose estimation into the pipeline
    #####
    # The reference to the file storing the previous predicted superquadric parameters
    suffix = img_dir.split("/")[-1]
    stored_stats_filename = "./grasp_pose_prediction/Marching_Primitives/sq_data/" + suffix + ".p"
    
    # Gripper Attributes
    gripper_width = 0.09 * nerf_scale
    gripper_length = 0.09 * nerf_scale
    gripper_thickness = 0.089 * nerf_scale
    gripper_attr = {"Type": "Parallel", "Length": gripper_length, \
                    "Width": gripper_width, "Thickness": gripper_thickness}

    # Obtain the gripper pose
    method_list = ["sq_split", "cg", "cg_depth"]
    for method in method_list:
        idx = 0
        avg_idx = 0
        mean_xyz = np.array([0.0, 0.0, 0.0])
        mean_dist = 0
        mean_num = 0
        for camera_pose_est in grasp_pose_cands[:1]: # NOTE: Change it to grasp_pose_cands for a complete review
            # Correct the frame conventions in camera & gripper of SPOT
            gripper_pose_current = camera_pose_est@np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
            
            # Assume that the gripper & camera are at the same pose
            
            if method == "sq_split":
                # Predict the grasp poses
                grasp_pose_options = \
                        Namespace(\
                            train=False, normalize = False, store=False, \
                                visualization=options.visualization)
                grasp_poses_world = predict_grasp_pose_sq(gripper_pose_current, \
                                    mesh, csv_filename, \
                                    [0.0, 1.0], stored_stats_filename, \
                                        gripper_attr, grasp_pose_options)
            else:
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
                pc_full_depth = None
                if method == "cg":
                    # Read the vertices of the whole mesh, as well as the 
                    # convert the scene into the real-world scale
                    pc_full = np.array(mesh_points)
                    pc_full /= nerf_scale
                    cg_idx = np.random.randint(low=0, high=pc_full.shape[0], size=100000)
                    pc_full = pc_full[cg_idx, :]
                    camera_extrinsics[0:3, 3] /= nerf_scale
                

                    pc_full = np.linalg.inv(camera_extrinsics)@(np.vstack((pc_full.T, np.ones(pc_full.shape[0]))))
                    pc_full = (pc_full[0:3, :]).T
                    
                    pc_colors = np.array(mesh.vertex_colors)
                    # Predict grasp poses based on the whole mesh
                    grasp_poses_cg, _ = predict_grasp_pose_contact_graspnet(\
                                pc_full, camera_intrinsics_matrix, pc_colors=pc_colors,\
                                filter_grasps=False, local_regions=False,\
                                mode = "xyz", visualization=options.visualization)
                elif method == "cg_depth":
                    # Obtain the depth array in instant-NGP's scale
                    depth_array = depth_map_mesh(mesh, camera_intrinsics_matrix, camera_extrinsics)
                    # Convert the value into the real-world scene
                    depth_array /= nerf_scale
                    camera_extrinsics[0:3, 3] /= nerf_scale

                    # Save the depth array for debugging purpose
                    # depth_array_save = depth_array * (65536/2)
                    # Image.fromarray(depth_array_save.astype('uint16')).save("./depth_test.png")
                    # Predict grasp poses on the Depth array (in real-world scale)
                    grasp_poses_cg, pc_full_depth = predict_grasp_pose_contact_graspnet(\
                                depth_array, camera_intrinsics_matrix, pc_colors=None,\
                                filter_grasps=False, local_regions=False,\
                                mode = "depth", visualization=options.visualization)
                    
                    # Convert pc_full, the depth point cloud, to world frame
                    pc_full_depth = pc_full_depth@camera_extrinsics[:3, :3].T + camera_extrinsics[0:3, 3].T
                    pc_full_depth *= nerf_scale
                
        
                # Transform the relative transformation into world frame for visualization purpose
                # (Now, they are relative transformations between frame "temp" (the camera) and 
                # predicted grasp poses)
                for i in range(grasp_poses_cg.shape[0]):
                    grasp_poses_cg[i] = camera_extrinsics@grasp_poses_cg[i]

                    # Convert all grasp poses back into instant-NGP's scale
                    # for visualization & antipodal test purpose
                    grasp_poses_cg[i][0:3, 3] *= nerf_scale
            
                    # Different gripper conventions in Contact GraspNet & SPOT
                    grasp_poses_cg[i] = grasp_poses_cg[i]@np.array([
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [1, 0, 0, 0.0584 * nerf_scale],
                        [0, 0, 0, 1]])
                
                camera_extrinsics[0:3, 3] *= nerf_scale
                # Evaluate the grasp poses based on antipodal & collision tests
                grasp_poses_world = grasp_pose_eval_gripper_cg(mesh, grasp_poses_cg, gripper_attr, \
                                camera_extrinsics, visualization = options.visualization, pc_full=pc_full_depth)

            tran_norm_min = np.Inf
            grasp_pose_gripper = np.eye(4)
            if len(grasp_poses_world) == 0:
                # If no valid grasp pose is predicted
                print("=======Results at Pose "+ str(idx) + " for " + method)
                print("No valid grasp pose is predicted...")
                idx += 1
                continue
            # Further filter out predicted poses to obtain the best one
            for grasp_pose in grasp_poses_world:
                # Correct the frame convention between the gripper of SPOT
                # & the one used in grasp pose prediction
                # In grasp pose prediction module, the gripper is pointing along negative x-axis
                grasp_pose = grasp_pose@np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

                # Evaluate the transformation between each predicted grasp pose
                # and the current gripper pose
                rot1 = grasp_pose[:3,:3]
                rot2 = gripper_pose_current[:3,:3]
                tran = rot2.T@rot1

                # Rotation along x-axis is symmetric
                r = R.from_matrix(tran[:3, :3])
                r_xyz = r.as_euler('xyz', degrees=True)

                # Rotation along x-axis is symmetric
                if r_xyz[0] > 90:
                    r_xyz[0] = r_xyz[0] - 180
                elif r_xyz[0] < -90:
                    r_xyz[0] = r_xyz[0]+ 180
                tran = R.from_euler('xyz', r_xyz, degrees=True).as_matrix()

                # Find the one with the minimum "distance"
                tran_norm = np.linalg.norm(tran - np.eye(3))
                if tran_norm < tran_norm_min:
                    tran_norm_min = tran_norm
                    grasp_pose_gripper = grasp_pose

            rel_transform_gripper = np.linalg.inv(gripper_pose_current)@grasp_pose_gripper
            ## Visualization platform
            if options.visualization:
                print("Visualize the final grasp result")
                # Create the window to display the grasp
                vis= o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(mesh)

                # Visualize the gripper as well
                arm_end = np.array([-gripper_length, 0, 0])
                center = np.array([0, 0, 0])
                elbow1 = np.array([0, 0, gripper_width/2])
                elbow2 = np.array([0, 0, -gripper_width/2])
                tip1 = np.array([gripper_length, 0, gripper_width/2])
                tip2 = np.array([gripper_length, 0, -gripper_width/2])

                # Construct the gripper
                gripper_points = np.array([
                    center,
                    arm_end,
                    elbow1,
                    elbow2,
                    tip1,
                    tip2
                ])
                gripper_lines = [
                    [1, 0],
                    [2, 3],
                    [2, 4],
                    [3, 5]
                ]
                gripper_start = o3d.geometry.TriangleMesh.create_coordinate_frame()
                gripper_start.scale(10/64 * nerf_scale, [0, 0, 0])
                gripper_start.transform(gripper_pose_current)

                grasp_pose_lineset_start = o3d.geometry.LineSet()
                grasp_pose_lineset_start.points = o3d.utility.Vector3dVector(gripper_points)
                grasp_pose_lineset_start.lines = o3d.utility.Vector2iVector(gripper_lines)
                grasp_pose_lineset_start.transform(gripper_pose_current)
                grasp_pose_lineset_start.paint_uniform_color((1, 0, 0))
                
                gripper_end = o3d.geometry.TriangleMesh.create_coordinate_frame()
                gripper_end.scale(10/64 * nerf_scale, [0, 0, 0])
                gripper_end.transform(grasp_pose_gripper)


                fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                fundamental_frame.scale(20/64, [0, 0, 0])

                grasp_pose_lineset_end = o3d.geometry.LineSet()
                grasp_pose_lineset_end.points = o3d.utility.Vector3dVector(gripper_points)
                grasp_pose_lineset_end.lines = o3d.utility.Vector2iVector(gripper_lines)
                grasp_pose_lineset_end.transform(grasp_pose_gripper)
                grasp_pose_lineset_end.paint_uniform_color((1, 0, 0))
                vis.add_geometry(gripper_start)
                vis.add_geometry(gripper_end)
                vis.add_geometry(grasp_pose_lineset_end)
                vis.add_geometry(grasp_pose_lineset_start)
                vis.add_geometry(fundamental_frame)
                vis.run()

                # Close all windows
                vis.destroy_window()
            # Print out the statistics
            r = R.from_matrix(rel_transform_gripper[:3, :3])
            print("======Valid Results at Pose " + str(idx) + " for " + method)
            print("======Num of Valid Grasp Poses====")
            print(len(grasp_poses_world))
            print("======Relative Distance =====")
            r_xyz = r.as_euler('xyz', degrees=True)
            # Rotation along x-axis is symmetric
            if r_xyz[0] > 90:
                r_xyz[0] = r_xyz[0] - 180
            elif r_xyz[0] < -90:
                r_xyz[0] = r_xyz[0]+ 180
            
            l = np.linalg.norm(rel_transform_gripper[0:3, 3])
            print(l)
            mean_xyz += r_xyz
            mean_dist += l
            mean_num += len(grasp_poses_world)
            idx += 1
            avg_idx += 1

        print("==== Average Results: " + method + "===")
        print(np.mean(abs(mean_xyz/avg_idx)))
        print(mean_dist/(nerf_scale * avg_idx))
        print(mean_num/idx)
        print(avg_idx)
            
