import numpy as np
import os


import sys
sys.path.append(os.getcwd())
pyngp_path = os.getcwd() + "/instant-ngp/build"
sys.path.append(pyngp_path)
import pyngp as ngp

import pyrender
import trimesh
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse
from grasp_pose_prediction.superquadrics import *
from utils.mesh_process import *
from utils.image_process import *

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

# Necessary Packages for sq parsing
from grasp_pose_prediction.Marching_Primitives.sq_split import sq_predict_mp
from grasp_pose_prediction.Marching_Primitives.MPS import add_mp_parameters
from grasp_pose_prediction.Marching_Primitives.mesh2sdf_convert import mesh2sdf_csv
'''
Main Program of the whole grasp pose prediction module
using Marching Primitives to split the target object into sq's
'''

def grasp_pose_eval_gripper(mesh, sq_closest, grasp_poses, gripper_attr, \
                            csv_filename, visualization = False):
    '''
    The function to evaluate the predicted grasp poses on the target mesh
    
    Input:
    sq_closest: the target superquadric (the closest superquadric to the camera)
    grasp_poses: predicted grasp poses based on the superquadrics
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
            # Find the grasp pose in the world frame (converted from sq local frame)
            grasp_pose = np.matmul(sq_closest["transformation"], grasp_pose)
            
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
            # collision_res = collision_test_sdf(csv_filename, gripper_points_vis_sample[:-1].T, threshold=0.05 * gripper_width)
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
                        # grasp_pose_mesh.paint_uniform_color((235/255, 197/255, 28/255))
                        # Color them into red instead (stricter)
                        grasp_pose_mesh.paint_uniform_color((1, 0, 0))
            if visualization:
                grasp_cands.append(grasp_pose_mesh)
    

    return bbox_cands, grasp_cands, grasp_poses_world

def lie_algebra(gripper_pose):
    '''
    Convert a SE(3) pose to se(3)
    '''
    translation = gripper_pose[0:3, 3]
    omega = R.from_matrix(gripper_pose[:3, :3]).as_rotvec()
    x, y, z = omega
    theta = np.linalg.norm(omega)
    omega_hat = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    coeff = 1 - (theta * np.cos(theta/2))/(2*np.sin(theta/2))
    V_inv = np.eye(3) - (1/2) * omega_hat + (coeff / theta ** 2) * (omega_hat@omega_hat)
    tp = V_inv@translation.flatten()
    vw = np.hstack((tp, omega))
    assert vw.shape[0] == 6
    return vw

def record_gripper_pose_sq(gripper_pose, sq_parameters, filename="./record.txt"):
    '''
    The function to record the gripper pose and the associated selected index of sq
    from the demos
    '''
    f = open(filename, "a")
    vw = lie_algebra(gripper_pose)
    sq_xyz = sq_parameters["location"]
    line = str(vw[0]) + ", " + str(vw[1]) + ", " + str(vw[2]) + ", " + \
        str(vw[3]) + ", " + str(vw[4]) + ", " + str(vw[5]) + ", " + \
        str(sq_xyz[0]) + ", " + str(sq_xyz[1]) + ", " + str(sq_xyz[2]) + "\n"
    f.write(line)
    f.close()

def predict_grasp_pose_sq(camera_pose, \
                          mesh, csv_filename, \
                          normalize_stats, stored_stats_filename, \
                            gripper_attr, args, point_select = None, region_select = False):
    '''
    Input:
    camera_pose: pose of the camera
    mesh: the mesh of the target object
    csv_filename: name of the file storing the corresponding csv values
    normalize_stats: stats in normalizing the mesh (used by mesh2sdf)
    stored_stats_filename: pre-stored stats of the splitted superquadrics
    gripper_attr: dict of the attributes of gripper
    args: user arguments

    Output:
    grasp_poses_camera: the grasp poses in the camera frame 
    Equivalently, the relative transformations between the camera and the grasp poses
    '''
    ##################
    ## Part I: Split the mesh into several superquadrics
    ##################
    ## Read the parameters of the superquadrics, or Re-calculate the splitting results
    # If the user hopes to re-obtain the results, repeat the splitting process
    if args.train: # If the user wants to reproduce the splitting process
        print("Splitting the Target Mesh (Marching Primitives)")
         # Split the target object into several primitives using Marching Primitives
        sq_predict = sq_predict_mp(csv_filename, args)
        # The normalization stats are never used, so just set them at the default values
        normalize_stats = [1.0, 0.0]
        sq_vertices_original, sq_transformation = read_sq_mp(\
            sq_predict, norm_scale=1.0, norm_d=0.0)
        if args.store:
            # If specified, store the statistics for the next use
            store_mp_parameters(stored_stats_filename, \
                        sq_vertices_original, sq_transformation, normalize_stats)
    else:
        # Try reading the sq parameters directly
        try:
            os.path.isfile(stored_stats_filename)
            print("Reading pre-stored Superquadric Parameters...")
            sq_vertices_original, sq_transformation, normalize_stats = read_mp_parameters(\
                                stored_stats_filename)
        except: 
            # If there is no pre-stored statistics, generate one
            print("Cannot find pre-stored Superquadric Splitting Results")
            print("Splitting the Target Mesh (Marching Primitives)")
            sq_predict = sq_predict_mp(csv_filename, args)
            if args.normalize:
                # Convert the predicted superquadrics back to the original scale
                sq_vertices_original, sq_transformation = read_sq_mp(\
                    sq_predict, normalize_stats[0], normalize_stats[1])
            else:
                normalize_stats = [1.0, 0.0]
                sq_vertices_original, sq_transformation = read_sq_mp(\
                    sq_predict, norm_scale=1.0, norm_d=0.0)
            if args.store:
                # If specified, store the statistics for the use next time
                store_mp_parameters(stored_stats_filename, \
                            sq_vertices_original, sq_transformation, normalize_stats)
    # Convert sq_verticies_original into a numpy array
    sq_vertices = np.array(sq_vertices_original).reshape(-1, 3)

    ## Find the sq associated to the selected point / the camera
    if point_select is None:
        camera_t = camera_pose[0:3, 3]
    else:
        camera_t = point_select
    sq_centers = []
    for val in sq_transformation:
        sq_center = val["transformation"][0:3 , 3]
        sq_centers.append(sq_center)
    sq_centers = np.array(sq_centers)

    # Compute the convex hull
    pc_sq_centers= o3d.geometry.PointCloud()
    pc_sq_centers.points = o3d.utility.Vector3dVector(sq_centers)
    hull, hull_indices = pc_sq_centers.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))

    # Optionally, let the user decide which superq region to grasp
    if region_select:
        if not args.visualization:
            print("To select a region to grasp, please activate the visualization functionality")
            return
        else:
            gui.Application.instance.initialize()

            window = gui.Application.instance.create_window("Graspable Region", 1024, 750)

            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(window.renderer)

            window.add_child(scene)

            scene.scene.add_geometry("target object", mesh, rendering.MaterialRecord())

            # Visualize the mesh and potentially graspable regions
            hull_idx = 0
            for sq_idx in hull_indices:
                sq_graspable = sq_transformation[sq_idx]["points"]
                sq_graspable_pc = o3d.geometry.PointCloud()
                sq_graspable_pc.points = o3d.utility.Vector3dVector(sq_graspable)
                sq_graspable_pc.paint_uniform_color((0, 0, 1))

                # Append an index near each sq region
                graspable_region_hull, _ = sq_graspable_pc.compute_convex_hull()
                graspable_region_hull.paint_uniform_color(np.random.rand(3))
                print(type(graspable_region_hull))
                print(type(mesh))
                scene.scene.add_geometry("graspable region " + str(hull_idx), graspable_region_hull, \
                                         rendering.MaterialRecord())

                scene.add_3d_label(sq_transformation[sq_idx]["transformation"][0:3, 3],\
                                   str(hull_idx))
                hull_idx = hull_idx + 1

            gui.Application.instance.run()  # Run until user closes window
            hull_select_idx = int(input("Which region you want to grasp?"))
        

    # Find the center of sq that is closest to the camera
    hull_vertices = np.array(hull_ls.points)
    hull_vertices_dist_idx = np.argsort(np.linalg.norm(hull_vertices - camera_t, axis=1))
    hull_v_idx = 0
    
    # Iteratively find the closest sq
    while True:
        if region_select:
            idx = hull_indices[hull_select_idx]
        else:
            idx = hull_indices[hull_vertices_dist_idx[hull_v_idx]]
        sq_closest = sq_transformation[idx]
        
        if args.visualization:
            print("================================")
            print("Selected superquadric Parameters: ")
            print(sq_closest["sq_parameters"])
            print("Index is: " + str(hull_vertices_dist_idx[hull_v_idx]))

        #######
        # Part II: Determine the grasp candidates on the selected sq and visualize them
        #######
        # Predict grasp poses around the target superquadric in LOCAL frame
        grasp_poses = grasp_pose_predict_sq_closest(sq_closest, gripper_attr, sample_number=50)
        # Evaluate the grasp poses w.r.t. the target mesh in WORLD frame
        bbox_cands, grasp_cands, grasp_poses_world = \
            grasp_pose_eval_gripper(mesh, sq_closest, grasp_poses, gripper_attr, \
                                    csv_filename, args.visualization)
        
        if len(grasp_poses_world) != 0: 
            # If a valid grasp pose is found
            print("Find one valid Grasp Pose!")

            # Write the camera/gripper pose and selected index to json file (DEBUG purpose)
            # gripper_pose = camera_pose@\
            #     np.array([[0, 0, 1, 0],
            #               [-1, 0, 0, 0],
            #               [0, -1, 0, 0],
            #               [0, 0, 0, 1]])
            # record_gripper_pose_sq(gripper_pose, sq_closest["sq_parameters"])
            break
        else: # If no good grasp pose is found, go to the next closest superquadric
            print("Failed to Find one valid Grasp Pose!")
            hull_v_idx += 1
            if hull_v_idx > 20: # Too many attempts
                print("Too many attempts...")
                break
            elif region_select:
                print("No valid grasp pose on the selected region")
                break
        
    ## Postlogue
    if args.visualization:
        # Optionally visualize the selected point
        ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        ball_select.scale(1/64, [0, 0, 0])

        ball_select.translate((camera_t[0], camera_t[1], camera_t[2]))
        ball_select.paint_uniform_color((1, 0, 0))
    
        # Delete the point cloud of the associated sq (to draw a new one; avoid point overlapping)
        sq_vertices_original.pop(idx)
        
        # Construct a point cloud representing the reconstructed object mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sq_vertices)
        # Visualize the super-ellipsoids
        pcd.paint_uniform_color((0.0, 0.5, 0))

        # Color the associated sq in blue and Complete the whole reconstructed model
        pcd_associated = o3d.geometry.PointCloud()
        pcd_associated.points = o3d.utility.Vector3dVector(sq_closest["points"])
        pcd_associated.paint_uniform_color((0, 0, 1))

        # Construct the mesh
        hull, _ = pcd_associated.compute_convex_hull()
        hull.paint_uniform_color((154/255, 229/255, 237/255))
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((0, 0, 1))
        pcd_associated = hull_ls
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
        camera_extrinsics = np.linalg.inv(camera_pose@\
                np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]]))
        camera_vis = o3d.geometry.LineSet.create_camera_visualization(\
            640, 480, camera_intrinsics, camera_extrinsics, scale=10/64)
        camera_vis.paint_uniform_color((1, 0, 0))
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(20/64, [0, 0, 0])
        camera_frame.transform(camera_pose@\
                np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]]))

        # Create the window to display everything
        vis= o3d.visualization.Visualizer()
        vis.create_window()
       
        
        vis.add_geometry(mesh)
        # vis.add_geometry(pcd)
        vis.add_geometry(pcd_associated) 
        # vis.add_geometry(sq_frame)
        # vis.add_geometry(frame)
        vis.add_geometry(camera_vis)
        vis.add_geometry(camera_frame)
        vis.add_geometry(ball_select)
        for grasp_cand in grasp_cands:
            vis.add_geometry(grasp_cand)
        for bbox_cand in bbox_cands:
            vis.add_geometry(bbox_cand)

        # ctr = vis.get_view_control()
        # # NOTE: the part to save a gif around the result
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

        #     vis.capture_screen_image('sq_split_screenshot/' + str(i) + '.png')
        # import imageio
        # images = []
        # for i in range(200):
        #     images.append(imageio.imread('sq_split_screenshot/' + str(i)+".png"))
        # imageio.mimsave('sq-split.gif', images, fps=5, loop=0)
        vis.run()

        # Close all windows
        vis.destroy_window()

        # Print out the validation results
        print("*******************")
        print("** Grasp pose Prediction Result: ")
        print("Selected Point in Space: ")
        print("Number of valid grasp poses predicted: " + str(len(grasp_poses_world)))
        print("*******************")

    return np.array(grasp_poses_world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives & predict grasp poses on top of them"
    )
    ## Arguments for Superquadrics Splitting Stuffs
    parser.add_argument(
        "nerf_dataset",
        help="The dataset containing all the training images & transform.json"
    )
    parser.add_argument(
        "--mesh_name",
        default = "target_obj.obj",
        help="The name of the mesh model to use"
    )
    # The snapshot of the trained NeRF model to use
    parser.add_argument(
        "--snapshot", default="base.ingp", type=str,
        help="Name of the snapshot of the NeRF model trained from instant-NGP"
    )
    # Arguments for camera
    parser.add_argument(
        '--camera', '-c',
        help='The name of the json file specifying the parameters\
            of the camera'
    )
    parser.add_argument(
        '--grid_resolution', type=int, default=100,
        help='Set the resolution of the voxel grids in the order of x, y, z, e.g. 64 means 100^3.'
    )

    parser.add_argument(
        '--level', type=float, default=2,
        help='Set watertighting thicken level. By default 2'
    ) 

    parser.add_argument(
        '--train', action = 'store_true', help="Whether to re-decompose the mesh into superquadrics"
    )
    parser.add_argument(
        '--store', action = 'store_true', help="Whether to store the re-decomposed result"
    )
    
    # Specify whether to select a region
    parser.add_argument(
        '--click', action = 'store_true', help="Whether to specify the region to grasp using a point on click"
    )
    parser.add_argument(
        '--region_select', action = 'store_true', help="Whether to specify a graspable region using index"
    )
    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true', help="Whether to visualize the grasp poses"
    )
    add_mp_parameters(parser)
    parser.set_defaults(normalize=False)
    parser.set_defaults(train=False)
    parser.set_defaults(store=True)
    parser.set_defaults(visualization=True)

    ######
    # Part 0: Read the mesh & the camera pose
    ######
    args = parser.parse_args(sys.argv[1:])
    ## The image used to specify the selected point
    nerf_dataset = args.nerf_dataset
    camera_dict = {}
    if args.camera is not None:
        f = open(os.path.join(args.nerf_dataset, args.camera))
        camera_dict = json.load(f)
        f.close()
    # Read the camera attributes
    fl_x = camera_dict["fl_x"]
    fl_y = camera_dict["fl_y"]
    cx = camera_dict["cx"]
    cy = camera_dict["cy"]
    if "camera_pose" in camera_dict:
        # If the user already specifies a pose, use it
        camera_pose = camera_dict["camera_pose"]
    else:
        # Otherwise, find a random pose
        dist = 1.5 * camera_dict["nerf_scale"]
        initial_pose = np.array([
            [0, 0, -1, dist],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        angle1 = -30*np.random.rand()
        angle2 = 360*np.random.rand()
        rotation = R.from_euler('zyz', [0, angle1, angle2], degrees=True).as_matrix()
        rotation = np.vstack((np.hstack((rotation, [[0], [0], [0]])), [0, 0, 0, 1]))
        camera_pose = rotation@initial_pose

    # In current case, the estimated camera pose is equal to the gt camera pose
    camera_pose_est = camera_pose
    ######
    # Part 1: Select a point at the image captured at the desired pose
    ######
    pos = None
    if args.click:
        # Construct the pyrender scene
        scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 1.0]))

        # Add the mesh
        tm = trimesh.load(os.path.join(nerf_dataset , args.mesh_name))
        m = pyrender.Mesh.from_trimesh(tm)
        scene.add(m)

        # Add the light source
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        light_initial = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2*dist],
            [0, 0, 0, 1]
        ])
        scene.add(light, pose=light_initial)

        # Add more light sources

        # Rotate around the x-axis for pi/2
        rot1 = R.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]).as_matrix()

        # Rotate around the z-axis for pi
        rot2 = R.from_quat([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)]).as_matrix()

        # Combine the two rotations
        rot = np.matmul(rot2, rot1)

        # Translation part
        d = np.array([[0], [2*dist], [0]])

        # Transformation matrix at the initial pose
        trans_initial = np.vstack((np.hstack((rot, d)), np.array([0, 0, 0, 1])))

        # More lighting sources
        for l in range(4):
            light = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=80.0)
            light_rot = R.from_quat([0, 0, np.sin(l * np.pi/4), np.cos(l * np.pi/4)]).as_matrix()
            light_tran = np.vstack((np.hstack((light_rot, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
            light_pose = np.matmul(light_tran, trans_initial)
            scene.add(light, pose=light_pose)
        # light = pyrender.light.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=500, name=None)
        # scene.add(light)
        # Add the camera
        camera = pyrender.IntrinsicsCamera(fl_x, fl_y, cx, cy)

        # Take a picture at the desired camera pose
        resolution = [camera_dict["w"], camera_dict["h"]]


        # Setting current transformation matrix (obey the convention in pyrender)
        camera_pose_est = np.matmul(camera_pose_est, \
                np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        nc = pyrender.Node(camera=camera, matrix=camera_pose_est)
        scene.add_node(nc)
        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)

        # Save the image to the folder
        grasp_image_name = os.path.join(nerf_dataset, "grasp_point.png")
        plt.imsave(grasp_image_name, \
                    color.copy(order='C'))
        
        # Restore the frame convention (in open3d)
        camera_pose_est = np.matmul(camera_pose_est, \
                np.array([[1, 0, 0, 0], 
                        [0, -1, 0, 0], 
                        [0, 0, -1, 0], 
                        [0, 0, 0, 1]]))  
        # camera_pose_est[0:3, 3] *= 0.33
        # camera_pose_est[0:3, 3] += 0.5
        # Select a point on the captured image at the specified pose
        image = cv2.imread(grasp_image_name)
        pick_x, pick_y = get_pick_vec_manual_force(image)
        print("================")
        print("Clicked Point")
        print(pick_x)
        print(pick_y)


        # Find the ray direction in camera frame
        initial_guess = [0, 0]

        # TODO: incorporate other distortion coefficients into concern as well
        camera_proj = np.array([
            [fl_x, 0, cx, 0],
            [0, fl_y, cy, 0],
            [0, 0, 1, 0]
        ])
        def equations(vars):
            x, y = vars
            eq = [
                camera_proj[0][0] * x + camera_proj[0][1] * y + camera_proj[0][2] * 1 - pick_x * 1,
                camera_proj[1][0] * x + camera_proj[1][1] * y + camera_proj[1][2] * 1 - pick_y * 1,
            ]
            return eq

        root = fsolve(equations, initial_guess)
        # Convert the point coorindate in world frame
        ray_dir = np.matmul(camera_pose_est, np.array([[root[0]], [root[1]], [1], [0]]))
        print("==Test Eq Solve")
        print(equations(root))
        print(root)
        print(camera_proj@np.array([[root[0]], [root[1]], [1], [0]]))
        # cv2.destroyAllWindows()

        # Normalize the directional vector
        print(ray_dir)
        ray_dir = ray_dir[0:3, 0]
        n = np.linalg.norm(ray_dir)
        ray_dir = ray_dir / n
        print(ray_dir)
        ray_proj_back = np.linalg.inv(camera_pose_est)@np.array([[ray_dir[0]], [ray_dir[1]], [ray_dir[2]], [0]])
        ray_proj_back *= n
        print(ray_proj_back)
        x,y,z = ray_proj_back[0][0], ray_proj_back[1][0], ray_proj_back[2][0]
        eq = [
                camera_proj[0][0] * x + camera_proj[0][1] * y + camera_proj[0][2] * z - pick_x * 1,
                camera_proj[1][0] * x + camera_proj[1][1] * y + camera_proj[1][2] * z - pick_y * 1,
            ]

        print(equations([x,y]))
    ##########
    # Part II: Read mesh & csv file (containing splitted superquadrics)
    # They should already be prepared from preprocess
    ##########

    # Read the mesh file
    filename= os.path.join(nerf_dataset , args.mesh_name)
    mesh = o3d.io.read_triangle_mesh(filename)
    # Read the csv file containing the sdf
    csv_filename = nerf_dataset + "/" + args.mesh_name[:-4] + ".csv"
    if args.click:
        # Find the coordinate of the selected point in space
        pos, dist  = point_select_in_space(camera_pose_est, ray_dir, mesh)
        print("========================")
        print("Selected Point in Space: ")
        print("[%.2f, %.2f, %.2f]"%(pos[0], pos[1], pos[2]))
        print(dist)
        dir_test = pos - camera_pose_est[0:3, 3]
        print(dir_test)
        print(dir_test / np.linalg.norm(dir_test))
    # The noramlization is never used, so just set it at the default value
    normalize_stats = [1.0, 0.0]
    ###############
    ## Part III: Predict Grasp poses
    ###############
    # The reference to the file storing the previous predicted superquadric parameters
    suffix = nerf_dataset.split("/")[-1]
    stored_stats_filename = "./grasp_pose_prediction/Marching_Primitives/sq_data/" + suffix + ".p"

    # Attributes of gripper
    nerf_scale = camera_dict["nerf_scale"]
    gripper_width = 0.09 * nerf_scale
    gripper_length = 0.09 * nerf_scale
    gripper_thickness = 0.089 * nerf_scale
    gripper_attr = {"Type": "Parallel", "Length": gripper_length, \
                    "Width": gripper_width, "Thickness": gripper_thickness}
        
    # Predict grasp poses
    gripper_pose = camera_pose_est@np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    predict_grasp_pose_sq(gripper_pose, \
                          mesh, csv_filename, \
                          normalize_stats, stored_stats_filename, gripper_attr, args, point_select=pos,\
                            region_select=args.region_select)