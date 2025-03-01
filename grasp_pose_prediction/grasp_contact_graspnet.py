import glob
import open3d as o3d
import os
import argparse
from argparse import Namespace
import torch
import numpy as np
from utils.mesh_process import *
from utils.image_process import *

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps
from contact_graspnet_pytorch.checkpoints import CheckpointIO 

import copy
def grasp_pose_eval_gripper_cg(mesh, grasp_poses, gripper_attr,\
                             camera_pose, visualization = False, pc_full=None):
    '''
    The function to evaluate the predicted grasp poses on the target mesh based on 
    Contact GraspNet
    
    Input:
    grasp_poses: predicted grasp poses based on the mesh based on Contact GraspNet
    gripper_attr: attributes of the gripper
    Output: 
    bbox_cands, grasp_cands: meshes used in open3d for visualization purpose
    grasp_pose: the VALID grasp poses in world frame 
    (frame convention: the gripper's arm is along the positive x direction;
    the gripper's opening is along the z direction)
    '''
    if gripper_attr["Type"] == "Parallel":
        ## For parallel grippers, evaluate it based on antipodal metrics
        # Extract the attributes of the gripper
        gripper_width = gripper_attr["Width"]
        gripper_length = gripper_attr["Length"]
        gripper_thickness = gripper_attr["Thickness"]

        # Key points on the gripper
        num_sample = 20
        arm_end = np.array([gripper_length, 0, 0])
        center = np.array([0, 0, 0])
        elbow1 = np.array([0, 0, gripper_width/2])
        elbow2 = np.array([0, 0, -gripper_width/2])
        tip1 = np.array([-gripper_length, 0, gripper_width/2])
        tip2 = np.array([-gripper_length, 0, -gripper_width/2])

        # Construct the gripper
        vis_width = 0.004
        arm = o3d.geometry.TriangleMesh.create_cylinder(radius=vis_width, height=gripper_length)
        arm_rot = np.array([  [0.0000000,  0.0000000,  1.0000000],
            [0.0000000,  1.0000000,  0.0000000],
            [-1.0000000,  0.0000000,  0.0000000]])
        arm.rotate(arm_rot)
        arm.translate((gripper_length/2, 0, 0))
        hand = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_width, height=vis_width)
        hand.translate((0, 0, -gripper_width/2))
        finger1 = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_length, height=vis_width)
        finger2 = o3d.geometry.TriangleMesh.create_box(width=vis_width, depth=gripper_length, height=vis_width)
        finger_rot = np.array([  [0.0000000,  0.0000000,  -1.0000000],
            [0.0000000,  1.0000000,  0.0000000],
            [1.0000000,  0.0000000,  0.0000000]])
        finger1.rotate(finger_rot)
        finger2.rotate(finger_rot)
        finger1.translate((-gripper_length/2, 0, 0))
        finger2.translate((-gripper_length/2, 0, 0))
        finger1.translate((0, 0, gripper_width/2 - gripper_length/2))
        finger2.translate((0, 0, -gripper_width/2 - gripper_length/2))

        gripper = arm
        gripper += hand
        gripper += finger1
        gripper += finger2
        ## Part I: collision test preparation
        # Sample several points on the gripper
        gripper_part1 = np.linspace(arm_end, center, num_sample)
        gripper_part2 = np.linspace(elbow1, tip1, num_sample)
        gripper_part3 = np.linspace(elbow2, tip2, num_sample)
        gripper_part4 = np.linspace(elbow1, elbow2, num_sample)
        gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))

        # Add the thickness
        gripper_point_sample1 = copy.deepcopy(gripper_points_sample)
        gripper_point_sample1[:, 1] = -gripper_thickness/2
        gripper_point_sample2 = copy.deepcopy(gripper_points_sample)
        gripper_point_sample2[:, 1] = gripper_thickness/2

        # Stack all points together (points for collision test)
        gripper_points_sample = np.vstack((gripper_points_sample, gripper_point_sample1, gripper_point_sample2))
        
        ## Part II: collision test & antipodal test
        print("Evaluating Grasp Qualities....")
        grasp_cands = [] # All the grasp candidates
        bbox_cands = [] # Closing region of the gripper
        grasp_poses_world = []
        # Construct the grasp poses at the specified locations,
        # and add them to the visualizer optionally
        for grasp_pose in grasp_poses:
            # The grasp poses are already in the world frame
            # However, the grasp poses are obeying SPOT's frame convention,
            # not the frame convention used by antipodal test => conversion needed
            grasp_pose = grasp_pose@np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            
            # Sample points for collision test
            gripper_points_vis_sample = np.vstack(\
                (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
            gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
            
            if visualization:
                # Transform the associated points for visualization or collision testing to the correct location
                grasp_pose_mesh = copy.deepcopy(gripper)
                grasp_pose_mesh = grasp_pose_mesh.rotate(R=grasp_pose[:3, :3], center=(0, 0, 0))
                grasp_pose_mesh = grasp_pose_mesh.translate(grasp_pose[0:3, 3])
                
            # Do the necessary testing jobs
            antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/6)
            # collision_res, _, _ = collision_test_local(mesh, gripper_points_sample, \
                            # grasp_pose, gripper_attr, 0.05 * gripper_width, scale = 1.5)
            collision_res = collision_test(mesh, gripper_points_vis_sample[:-1].T, threshold=0.03 * gripper_width)

            # Collision Test
            if collision_res:
                if visualization:
                    grasp_pose_mesh.paint_uniform_color((1, 0, 0))
            else: # Antipodal test
                if antipodal_res == True:
                    grasp_poses_world.append(grasp_pose)
                    if visualization:
                        bbox_cands.append(bbox)
                        grasp_pose_mesh.paint_uniform_color((0, 1, 0))
                else:
                    if visualization:
                         # Color them into yellow (no collision, but still invalid)
                        grasp_pose_mesh.paint_uniform_color((235/255, 197/255, 28/255))
                        # Color them into red instead (stricter)
                        # grasp_pose_mesh.paint_uniform_color((1, 0, 0))
            if visualization:
                grasp_cands.append(grasp_pose_mesh)
        
        # Visualize all grasp poses
        if visualization:
            vis= o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            # Plot out the fundamental frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            frame.scale(20/64, [0, 0, 0])

            # Plot out the camera frame
            fl_x = 552.0291012161067
            fl_y = 552.0291012161067
            cx = 320
            cy = 240
            camera_intrinsics = np.array([
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1]
            ])
            camera_extrinsics = np.linalg.inv(camera_pose)
            camera_vis = o3d.geometry.LineSet.create_camera_visualization(\
                640, 480, camera_intrinsics, camera_extrinsics, scale=10/64)
            camera_vis.paint_uniform_color((1, 0, 0))
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            camera_frame.scale(20/64, [0, 0, 0])
            camera_frame.transform(camera_pose)
            # vis.add_geometry(frame)
            vis.add_geometry(camera_frame)
            vis.add_geometry(camera_vis)
            for grasp_cand in grasp_cands:
                vis.add_geometry(grasp_cand)
            # for bbox_cand in bbox_cands:
            #     vis.add_geometry(bbox_cand)

            # Also, add the partial depth point cloud if desired
            if pc_full is not None:
                depth_pc = o3d.geometry.PointCloud()
                depth_pc.points = o3d.utility.Vector3dVector(pc_full)
                depth_pc.paint_uniform_color((0, 0, 1))
                vis.add_geometry(depth_pc)
            # TODO: remove this part
            # ctr = vis.get_view_control()
            # # TODO: remove this part
            # x = 150
            # y = -500
            # ctr.rotate(0, y, xo=0.0, yo=0.0)
            # ctr.rotate(x, 0, xo=0.0, yo=0.0)
            # ctr.translate(0, 0, xo=0.0, yo=0.0)
            # ctr.scale(0.005)
            # for i in range(200):
            #     x = -10
            #     y = 0
            #     ctr.rotate(x, y, xo=0.0, yo=0.0)
                
            #     # Updates
            #     # vis.update_geometry(pcd)
            #     # vis.update_geometry(mesh)
            #     # vis.update_geometry(camera_frame)
            #     vis.poll_events()
            #     vis.update_renderer()

            #     # Capture image
            #     if pc_full is not None:
            #         vis.capture_screen_image('cg_depth_screenshot/' + str(i) + '.png')
            #     else:
            #         vis.capture_screen_image('cg_screenshot/' + str(i) + '.png')
            # import imageio
            # images = []
            # if pc_full is not None:
            #     for i in range(200):
            #         images.append(imageio.imread('cg_depth_screenshot/' + str(i)+".png"))
            #     imageio.mimsave('cg-depth.gif', images, fps=5, loop=0)
            # else:
            #     for i in range(200):
            #         images.append(imageio.imread('cg_screenshot/' + str(i)+".png"))
            #     imageio.mimsave('cg.gif', images, fps=5, loop=0)
                
            vis.run()

            # Close all windows
            vis.destroy_window()

    return grasp_poses_world

def predict_grasp_pose_contact_graspnet(input_data,
              camera_intrinsics_matrix,
              pc_colors = None,
              mode = "depth",
              local_regions=True, 
              filter_grasps=True, 
              skip_border_objects=False,
              z_range = [0.2,2],
              forward_passes = 5,
              visualization = False):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    # Read the configuration
    FLAGS = Namespace(ckpt_dir ="./contact_graspnet_pytorch/checkpoints/contact_graspnet", forward_passes=5, arg_configs=[])
    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    # Build the model
    print("Building up Contact GraspNet model")
    grasp_estimator = GraspEstimator(global_config)

    # Load the weights
    model_checkpoint_dir = os.path.join(FLAGS.ckpt_dir, 'checkpoints')
    print(model_checkpoint_dir)
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        print('No model checkpoint found')
        load_dict = {}

    
    os.makedirs('./contact_graspnet_results', exist_ok=True)

    # Process example test scenes
    print("Loading Data...")
    if mode == "depth":
        depth = input_data
        cam_K = camera_intrinsics_matrix
        segmap = None
        rgb = None
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(\
            depth, cam_K, segmap=segmap, rgb=rgb, \
                skip_border_objects=skip_border_objects, z_range=z_range)
        pc_colors = np.ones((pc_full.shape[0], 3))
        if pc_full.shape[0] == 0:
            return pc_full # Failed to generate depth data...
    elif mode == "xyz":
        pc_full = input_data
        pc_colors = pc_colors *255
        cam_K = camera_intrinsics_matrix
        segmap = None
        rgb = None
        pc_segments = None

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(pc_full, 
                                                                                       pc_segments=pc_segments, 
                                                                                       local_regions=local_regions, 
                                                                                       filter_grasps=filter_grasps, 
                                                                                       forward_passes=forward_passes)  
    if visualization:
        # Visualize results          
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    num = min(50, scores[-1].shape[0])
    return pred_grasps_cam[-1][np.argpartition(scores[-1], -num)[-num:]], pc_full


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/contact_graspnet', help='Log dir')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    
    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, 
              FLAGS.ckpt_dir,
              FLAGS.np_path, 
              local_regions=FLAGS.local_regions,
              filter_grasps=FLAGS.filter_grasps,
              skip_border_objects=FLAGS.skip_border_objects,
              z_range=eval(str(FLAGS.z_range)),
              forward_passes=FLAGS.forward_passes,
              K=eval(str(FLAGS.K)))