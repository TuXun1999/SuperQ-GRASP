import os, sys
import argparse
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
from utils.mesh_process import coordinate_correction
from grasp_pose_prediction.superquadrics import *
from utils.mesh_process import *
from utils.image_process import *
# Necessary Packages for sq parsing
from grasp_pose_prediction.Marching_Primitives.mesh2sdf_convert import mesh2sdf_csv
from grasp_pose_prediction.Marching_Primitives.MPS import add_mp_parameters
from grasp_pose_prediction.Marching_Primitives.sq_split import sq_predict_mp

import open3d as o3d
def instant_NGP_screenshot(nerf_dataset, snapshot, camera_intrinsics_dict,\
                           camera_pose, mode = "Depth"):
    # Construct the instant-NGP testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    #Loading Snapshot
    testbed.load_snapshot(os.path.join(nerf_dataset, snapshot))

    resolution = [camera_intrinsics_dict["w"], camera_intrinsics_dict["h"]]
    fov_x = camera_intrinsics_dict["camera_angle_x"] * 180/np.pi
    fov_y = camera_intrinsics_dict["camera_angle_y"] * 180/np.pi
    testbed.fov = max(fov_x, fov_y)

    if mode == "RGB":
        testbed.render_mode = ngp.Shade

        # Setting current transformation matrix.
        testbed.set_nerf_camera_matrix(camera_pose[:-1])
        # Formally take a picture at that pose
        frame = testbed.render(resolution[0],resolution[1],8,linear=False)
        
    elif mode == "Depth":
        testbed.render_mode = ngp.Depth

        # Setting current transformation matrix
        testbed.set_nerf_camera_matrix(camera_pose[:-1])
        # Formally take a picture at that pose
        frame = testbed.render(resolution[0], resolution[1], 8, linear=True)
        frame = frame[:, :, 0]

    return frame
def preprocess(camera_intrinsics_dict, dist, nerf_dataset, snapshot, options):
    '''
    The function to preprocess the reconstructed NeRF model
    Input:
    camera_intrinsics_dict: dictionary of the intrinsics of the camera to use
    dist: distance of the surrounding cameras
    nerf_dataset: the directory to the stored NeRF model

    Output:
    Several files under the SAME directory of the stored NeRF model => for future usage
    including:
    1. a mesh model => pose estimation & grasp pose prediction
    2. SDF value generated from the mesh model
    3. several sample images => poses of them are already known; used for pose estimation
    '''
    mesh_filename = nerf_dataset + "/target_obj.obj"
    ## Load the snapshot of the NeRF training process
    if snapshot is not None:
        # If the mesh comes from instant-NGP, load the snapshot and generate
        # the mesh

        # Construct the instant-NGP testbed
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

        #Loading Snapshot
        testbed.load_snapshot(os.path.join(nerf_dataset, snapshot))
        

        ## Save the mesh
        res = options.marching_cubes_res or 256
        thresh = options.marching_cubes_density_thresh or 2.5
        print(f"Generating mesh via marching cubes and saving to \
            {mesh_filename}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
        testbed.compute_and_save_marching_cubes_mesh(\
            mesh_filename, [res, res, res], thresh=thresh)
        # Correct the order
        # In instant-NGP, the data order is yzx, not xyz

        # Read the file as a triangular mesh again
        mesh = o3d.io.read_triangle_mesh(mesh_filename)

        # Smooth the mesh with Taubin
        l = options.taubin_smooth
        print('filter with Taubin with ' + str(l) + ' iterations')
        mesh = mesh.filter_smooth_taubin(number_of_iterations=l)
        mesh = coordinate_correction(mesh, mesh_filename)
    else:
        # If the mesh is already given from a dataset, add vertices to it
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        # A gt mesh doesn't need Taubin smoothing
        
        # Correct the coordinate (z->upward)
        mesh = coordinate_correction(mesh, mesh_filename)
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        o3d.io.write_triangle_mesh(mesh_filename, mesh)


    # Also prepare the SDF
    if options.sdf_normalize:
        csv_filename = nerf_dataset + "/target_obj_normalized.csv"
    else:
        csv_filename = nerf_dataset + "/target_obj.csv"

    # Read the csv file containing sdf value
    if options.normalize:
        # If the user wants a normalized model, generate the sdf anyway
        normalize_stats = mesh2sdf_csv(mesh_filename, options)
    else: 
        # If normalization is not desired, the stats will always be [1.0, 0.0]
        normalize_stats = [1.0, 0.0]
        print("Converting mesh into SDF...")
        # If the csv file has not been generated, generate one
        normalize_stats = mesh2sdf_csv(mesh_filename, options)
    
    # Split the mesh into primitives offline in advance
    if not options.no_mesh_split: # Split the mesh
        print("Splitting the Target Mesh (Marching Primitives)")
        # Split the target object into several primitives using Marching Primitives
        sq_predict = sq_predict_mp(csv_filename, options)
        normalize_stats = [1.0, 0.0]
        sq_vertices_original, sq_transformation = read_sq_mp(\
            sq_predict, norm_scale=1.0, norm_d=0.0)
        
        # Store the results in the local folder
        suffix = nerf_dataset.split("/")[-1]
        stored_stats_filename = "./grasp_pose_prediction/Marching_Primitives/sq_data/" + suffix + ".p"
        store_mp_parameters(stored_stats_filename, \
                        sq_vertices_original, sq_transformation, normalize_stats)
    ## Save the generated images on the target object
    if snapshot is not None and options.image_screenshot_nerf:
        # Path to folder to save rendered photos in.
        images_folder = os.path.join(nerf_dataset,"images")
        
        # Delete the folder if it exists.
        if os.path.exists(images_folder) == True:
            shutil.rmtree(images_folder)

        # Intialize the folder.
        os.makedirs(images_folder)

        # Extract out camera information
        result_dict = copy.deepcopy(camera_intrinsics_dict)
        result_dict["frames"] = []
        if camera_intrinsics_dict.get("nerf_scale") is None:
            print("To go further in real-world experiments, a scale between real-world scene and the NeRF scene should be given")
            print("(The value used to scale the real scene into the scene contained in an unit cube used by instant-NGP)")
            sys.exit(1)
        result_dict["nerf_scale"] = camera_intrinsics_dict["nerf_scale"]

        resolution = [camera_intrinsics_dict["w"], camera_intrinsics_dict["h"]]
        fov_x = camera_intrinsics_dict["camera_angle_x"] * 180/np.pi
        fov_y = camera_intrinsics_dict["camera_angle_y"] * 180/np.pi
        testbed.fov = max(fov_x, fov_y)

        
        testbed.render_mode = ngp.Shade
        # Initialize a json file, just as is usually shown in a NeRF dataset used in instant-NGP
        json_file = open(nerf_dataset + "/transforms.json", "w")

        # Initialize the starting pose
        # Notice that for instant-NGP, the starting pose is
        # x-> forward, y->leftward, z->downward

        # Rotate around the x-axis for pi/2
        rot1 = R.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]).as_matrix()

        # Rotate around the z-axis for pi
        rot2 = R.from_quat([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)]).as_matrix()

        # Combine the two rotations
        rot = np.matmul(rot2, rot1)

        
        # Number of circles 
        c_num = 1
        # Number of samples vertically
        h_num = 6
        angle_h_start = 0
        angle_dev_h = np.pi/(3*h_num)
        # Number of samples along the circle around the object
        i_num = 24
        angle_dev_i = 2*np.pi/i_num

        angle_i_start = 0

        for c in range(0, c_num):
            # Translation part
            d = np.array([[0], [dist + c], [0]])

            # Transformation matrix at the initial pose
            trans_initial = np.vstack((np.hstack((rot, d)), np.array([0, 0, 0, 1])))
        
            # Initialize all the camera view pose candidates & take the picture
            for h in range(0, h_num): # Rotate around axis-x
                rot_angle_h = angle_h_start + h * angle_dev_h
                # Rotation matrix for the camera pose around x-axis
                rot_around_x = R.from_quat([np.sin(rot_angle_h/2), 0, 0, np.cos(rot_angle_h/2)]).as_matrix()
                camera_pose_vet = np.matmul(
                        np.vstack(
                            (np.hstack((rot_around_x, np.array([[0], [0], [0]]))), 
                            np.array([0, 0, 0, 1]))),
                        trans_initial
                        )
                # camera_pose_vet = np.matmul(
                #         np.array([
                #             [1, 0, 0, 0],
                #             [0, 1, 0, 0],
                #             [0, 0, 1, 0.5 + 0.4/h_num * h],
                #             [0, 0, 0, 1]
                #         ]),
                #         trans_initial
                #         )
                for i in range(i_num): # Rotate around axis-z
                    rot_angle_i  = angle_i_start + i * angle_dev_i

                    # Rotation matrix for that camera pose
                    rot_around_z = R.from_quat([0, 0, np.sin(rot_angle_i/2), np.cos(rot_angle_i/2)]).as_matrix()

                    # Obtain the camera pose
                    camera_pose = np.matmul(
                        np.vstack(
                            (np.hstack((rot_around_z, np.array([[0], [0], [0]]))), 
                            np.array([0, 0, 0, 1]))),
                        camera_pose_vet
                        )

                    # Build up the camera frame
                    camera_frame = {}
                    camera_frame["file_path"] = "./images/" + str(c) + "_" + str(h) + "_" + str(i) + ".png"
                    camera_frame["transform_matrix"] = camera_pose.tolist()
                    result_dict["frames"].append(camera_frame)

                    # Setting current transformation matrix.
                    testbed.set_nerf_camera_matrix(camera_pose[:-1])
                    # Formally take a picture at that pose
                    frame = testbed.render(resolution[0],resolution[1],8,linear=False)

                    # Save the image to the folder
                    plt.imsave(images_folder + "/" + str(c) + "_" + str(h) + "_" + str(i) + ".png", \
                            frame.copy(order='C'))

        # Dump the dict to the json file
        json.dump(result_dict, json_file, indent=4)


    return normalize_stats, csv_filename

if __name__ == "__main__":
    """Command line interface."""
    parser = argparse.ArgumentParser()

    # Arguments for the NeRF model to use
    parser.add_argument(
        "nerf_model_directory",
        help="The directory containing the NeRF model to use"
    )
    parser.add_argument(
        "--snapshot", type=str,
        help="Name of the snapshot of the NeRF model trained from instant-NGP"
    )

    # Arguments for Mesh
    parser.add_argument("--marching_cubes_res", \
                        default=256, type=int, \
                            help="Sets the resolution for the marching cubes grid.")
    parser.add_argument("--marching_cubes_density_thresh", \
                     default=2.5, type=float, \
                        help="Sets the density threshold for marching cubes.")
    
    parser.add_argument(
        '--grid_resolution', type=int, default=100,
        help='Set the resolution of the voxel grids in the order of x, y, z, e.g. 64 means 100^3.'
    )

    parser.add_argument(   
        '--sdf-normalize', action='store_true',
        help='Whether to normalize the input mesh when generating SDF before feeding it to Primitive Splitting'
    )

    parser.add_argument(
        '--level', type=float, default=2,
        help='Set watertighting thicken level. By default 2'
    )
    parser.add_argument(
        '--taubin-smooth', '-t', type=int, default=50,
        help='Specify the Taubin smooth iterations to reduce noise'
    )
    parser.add_argument(
        '--no-mesh-split', '-m', action = 'store_true', help="Disable Mesh splitting"
    )
    parser.add_argument(
        '--image-screenshot-nerf', '-i', action='store_true', \
        help="(Optional) Take screenshots of the target object in NeRF to generate images at ground-truth poses.\
            (You can also use the images to train NeRF as well as the poses, as long as the camera\
                intrinsics are matched. Then, no need to call this option)"
    )
    add_mp_parameters(parser)
    parser.set_defaults(normalize=False)
    parser.set_defaults(no_mesh_split=False)
    parser.set_defaults(image_screenshot_nerf=False)

    # Arguments for pictures & pose estimation
    parser.add_argument(
        '--camera', '-c',
        help='The name of the json file specifying the intrinsic parameters\
            of the camera to use in real-world experiments'
    )
    parser.add_argument('--distance', \
                        type=float, default=1.5, \
                        help="Approximate Distance between the center of the \
                            target object & the camera. Used in pose estimation")
    options = parser.parse_args(sys.argv[1:])

    nerf_dataset = options.nerf_model_directory
    camera_intrinsics_dict = {}
    if options.camera is not None:
        f = open(os.path.join(options.nerf_model_directory, options.camera))
        camera_intrinsics_dict = json.load(f)
        f.close()

    ## Preprocess the given NeRF model
    #
    sdf_normalize_stats, csv_filename = preprocess(\
        camera_intrinsics_dict, options.distance, nerf_dataset, \
               options.snapshot, options)
    
    
