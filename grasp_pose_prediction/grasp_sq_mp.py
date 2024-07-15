import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import open3d as o3d

import argparse
from superquadrics import *
from mesh_process import *
from image_process import *

from scipy.spatial.transform import Rotation as R

# Necessary Packages for sq parsing
from grasp_pose_prediction.Marching_Primitives.sq_split import sq_predict_mp
from grasp_pose_prediction.Marching_Primitives.MPS import add_mp_parameters
from grasp_pose_prediction.Marching_Primitives.mesh2sdf_convert import mesh2sdf_csv
'''
Main Program of the whole grasp pose prediction module
using Marching Primitives to split the target object into sq's
'''

# TODO: Add argument parser here
def predict_grasp_pose_sq(camera_pose, nerf_scale, \
                          mesh, csv_filename, \
                          normalize_stats, stored_stats_filename, args):
    '''
    Input:
    camera_pose: pose of the camera
    nerf_scale: scaling factor between NeRF & real-world camera poses
    mesh: the mesh of the target object
    csv_filename: name of the file storing the corresponding csv values
    normalize_stats: stats in normalizing the mesh
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
        # Read the attributes of the predicted sq's
        if args.normalize: # normalize_stats is always defined if args.normalize is true
            # Convert the primitives back to the original scale
            sq_vertices_original, sq_transformation = read_sq_mp(\
                sq_predict, normalize_stats[0], normalize_stats[1])
        else:
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

    ## Find the sq associated to the selected point
    # New method: find the sq, the center of which is closest to the camera
    camera_t = camera_pose[0:3, 3]
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

    # Find the center of sq that is closest to the camera
    hull_vertices = np.array(hull_ls.points)
    hull_v_idx = np.argmin(np.linalg.norm(hull_vertices - camera_t, axis=1))
    idx = hull_indices[hull_v_idx]

    sq_closest = sq_transformation[idx]
    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    ball_select.translate((sq_centers[idx][0], sq_centers[idx][1], sq_centers[idx][2]))
    ball_select.paint_uniform_color((1, 0, 0))
    # Delete the point cloud of the associated sq (to draw a new one; avoid point overlapping)
    sq_vertices_original.pop(idx)
    
    # Construct a point cloud representing the reconstructed object mesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sq_vertices)
    # Visualize the super-ellipsoids
    pcd.paint_uniform_color((0.0, 0.4, 0))

    # Color the associated sq in blue and Complete the whole reconstructed model
    pcd_associated = o3d.geometry.PointCloud()
    pcd_associated.points = o3d.utility.Vector3dVector(sq_closest["points"])
    pcd_associated.paint_uniform_color((0, 0, 1))
    
    print("================================")
    print("Selected superquadric Parameters: ")
    print(sq_closest["sq_parameters"])

    #######
    # Part II: Determine the grasp candidates on the selected sq and visualize them
    #######
    # Define the gripper
    # From experiments, when nerf_scale is 64, the gripper of length 10 looks
    # pretty nice
    scale = 5 
    gripper_width = 2 * scale * (nerf_scale/ 64)
    gripper_length = 2 * scale * (nerf_scale/ 64)
    gripper_thickness = 0.4 * gripper_length

    # Key points on the gripper
    num_sample = 20
    arm_end = np.array([gripper_length, 0, 0])
    center = np.array([0, 0, 0])
    elbow1 = np.array([0, 0, gripper_width/2])
    elbow2 = np.array([0, 0, -gripper_width/2])
    tip1 = np.array([-gripper_length, 0, gripper_width/2])
    tip2 = np.array([-gripper_length, 0, -gripper_width/2])

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

    # Stack all points together
    gripper_points_sample = np.vstack((gripper_points_sample, gripper_point_sample1, gripper_point_sample2))


    # Pass the gripper attributes to predict the grasp poses
    gripper_attr = {"Type": "parallel", \
                    "Width": gripper_width, "Length": gripper_length, "Thickness": gripper_thickness,\
                        "Scale": scale}
    grasp_poses = grasp_pose_predict_sq_closest(sq_closest, gripper_attr, sample_number=40)
    


    print("Evaluating Grasp Qualities....")
    grasp_cands = [] # All the grasp candidates
    bbox_cands = [] # Closing region of the gripper
    grasp_poses_camera = []
    # Construct the grasp poses at the specified locations,
    # and add them to the visualizer
    for grasp_pose in grasp_poses:
        # Find the grasp pose in the world frame (converted from sq local frame)
        grasp_pose = np.matmul(sq_closest["transformation"], grasp_pose)
       
        
        # Transform the associated points for visualization or collision testing to the correct location
        gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
        gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
        gripper_points_vis_sample = np.vstack(\
            (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
        gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
        
        grasp_pose_lineset = o3d.geometry.LineSet()
        grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
        grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
        
        # Do the necessary testing jobs
        # grasp_pose_pc = o3d.geometry.PointCloud()
        # grasp_pose_pc.points = o3d.utility.Vector3dVector(gripper_points_vis_sample[:-1].T)
        # grasp_pose_pc.paint_uniform_color((1, 0, 0))
        antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/9)
        # collision_res, _, _ = collision_test_local(mesh, gripper_points_sample, \
                        # grasp_pose, gripper_attr, 0.05 * gripper_width, scale = 1.5)
        collision_res = collision_test(mesh, gripper_points_vis_sample[:-1].T, threshold=0.05 * gripper_width)
        # Collision Test
        if collision_res:
            grasp_pose_lineset.paint_uniform_color((1, 0, 0))
        else:
            if antipodal_res == True:
                bbox_cands.append(bbox)
                grasp_pose_lineset.paint_uniform_color((0, 1, 0))
                grasp_poses_camera.append(grasp_pose@np.linalg.inv(camera_pose))
            else:
                grasp_pose_lineset.paint_uniform_color((1, 1, 0))
        grasp_cands.append(grasp_pose_lineset)
        
    ## Postlogue
    # Plot out the fundamental frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(20/64 * nerf_scale, [0, 0, 0])

    # Plot out the camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame.transform(camera_pose)


    # Create the window to display everything
    vis= o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_associated) 
    # vis.add_geometry(sq_frame)
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(ball_select)
    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)
    for bbox_cand in bbox_cands:
        vis.add_geometry(bbox_cand)
    vis.run()

    # Close all windows
    vis.destroy_window()

    # Print out the validation results
    print("*******************")
    print("** Grasp pose Prediction Result: ")
    print("Selected Point in Space: ")
    print("Number of valid grasp poses predicted: " + str(len(bbox_cands)))
    print("*******************")

    return np.array(grasp_poses_camera)


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
        "--image_name",
        help="The name of the image to use for point selection"
    )
    parser.add_argument(
        "--mesh_name",
        default = "chair_upper.obj",
        help="The name of the mesh model to use"
    )

    parser.add_argument(
        '--grid_resolution', type=int, default=100,
        help='Set the resolution of the voxel grids in the order of x, y, z, e.g. 64 means 100^3.'
    )

    parser.add_argument(   
        '--normalize', action='store_true',
        help='Whether to normalize the input mesh when generating SDF before feeding it to Primitive Splitting'
    )

    parser.add_argument(
        '--level', type=float, default=2,
        help='Set watertighting thicken level. By default 2'
    )
    parser.add_argument(
        '--train', action = 'store_true'
    )
    parser.add_argument(
        '--store', action = 'store_true'
    )

    parser.add_argument(
        '--laplacian-smooth', '-l', type=int, default=50,
        help='Specify the laplacian smooth iterations to reduce noise'
    )

    add_mp_parameters(parser)
    parser.set_defaults(normalize=False)
    parser.set_defaults(train=False)
    parser.set_defaults(store=True)

    ######
    # Part 0: Read the mesh & the camera poses
    ######
    args = parser.parse_args(sys.argv[1:])
    ## The image used to specify the selected point
    img_dir = args.nerf_dataset

    # If the image is not specified, select one image by random
    if args.image_name is None:
        image_files = os.listdir(args.nerf_dataset + "/images")
        image_idx = np.random.randint(0, len(image_files))
        image_name = image_files[image_idx]
    else:
        image_name = args.image_name


    img_file = "/images/" + image_name
    print("====================")
    print("Select from Image")
    print(img_file)
    ## Obtain the ray direction of the selected point in space 
    # NOTE: Previous method to select the grasp point manually
    # ray_dir, camera_pose, _, nerf_scale = point_select_from_image(img_dir, img_file, save_fig=True)
    # ray_dir = ray_dir / np.linalg.norm(ray_dir)
    _, camera_pose,nerf_scale = read_proj_from_json(img_dir, img_file)
    ## Create files as input to other modules
    # Specify the mesh file
    filename=img_dir + "/" + args.mesh_name
    
    # Read the file as a triangular mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    l = args.laplacian_smooth
    print('filter with Laplacian with ' + str(l) + ' iterations')
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=l)


    ##########
    # Part I: Convert the mesh into csv file
    ##########

    ## Read the csv file containing the sdf
    if args.normalize:
        csv_filename = img_dir + "/" + args.mesh_name[:-4] + "_normalized.csv"
    else:
        csv_filename = img_dir + "/" + args.mesh_name[:-4] + ".csv"

    # Determine whether to correct the coordinate convention (based on whether csv is
    # generated; if so, it means that the coordinate correction has already been done)
    if not os.path.isfile(csv_filename):
        # Restore the correct scale & fix up the coordinate issue 
        mesh = coordinate_correction(mesh, filename, nerf_scale)
        print('filter with Laplacian with ' + str(l) + ' iterations')
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=l)

    # Read the csv file containing sdf value
    if args.normalize:
        # If the user wants a normalized model, generate the sdf anyway
        normalize_stats = mesh2sdf_csv(filename, args)
    else: 
        # If normalization is not desired, the stats will always be [1.0, 0.0]
        normalize_stats = [1.0, 0.0]
        if os.path.isfile(csv_filename):
            # If not, try to read the sdf in a pre-stored csv file directly
            print("Reading SDF from csv file: ")
            print(csv_filename)
        else:
            print("Converting mesh into SDF...")
            # If the csv file has not been generated, generate one
            normalize_stats = mesh2sdf_csv(filename, args)

    ###############
    ## Part II: Split the mesh into several primitives & Predict Grasp poses
    ###############
    # The reference to the file storing the previous predicted superquadric parameters
    suffix = img_dir.split("/")[-1]
    stored_stats_filename = "./grasp_pose_prediction/Marching_Primitives/sq_data/" + suffix + ".p"
    predict_grasp_pose_sq(camera_pose, nerf_scale, \
                          mesh, csv_filename, \
                          normalize_stats, stored_stats_filename, args)