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
def preprocess(camera_intrinsics_dict, dist, nerf_dataset, snapshot, options):
    '''
    The function to preprocess the reconstructed NeRF model
    Input:
    camera_intrinsics_dict: dictionary of the intrinsics of the camera to use
    dist: distance of the surrounding cameras
    nerf_dataset: the directory to the stored NeRF model
    nerf_scale: the scaled used to fit the real scene into the unit cube used in instant-NGP

    Output:
    Several files under the SAME directory of the stored NeRF model => for future usage
    including:
    1. a mesh model => pose estimation & grasp pose prediction
    2. several sample images => poses of them are already known; used for pose estimation
    '''
    ## Load the snapshot of the NeRF training process
    # Construct the instant-NGP testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    #Loading Snapshot
    testbed.load_snapshot(os.path.join(nerf_dataset, snapshot))
    

    ## Save the mesh
    res = options.marching_cubes_res or 256
    thresh = options.marching_cubes_density_thresh or 2.5
    mesh_filename = nerf_dataset + "/target_obj.obj"
    print(f"Generating mesh via marching cubes and saving to \
          {mesh_filename}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
    testbed.compute_and_save_marching_cubes_mesh(\
        mesh_filename, [res, res, res], thresh=thresh)
    # Correct the order & scale in mesh file
    # (When loaded into NeRF / instant-NGP, the scene may be re-scaled to fit the unit cube;
    # also, by default, the data order is yzx, not xyz)
    # Restore the correct scale & fix up the coordinate issue 

    # Read the file as a triangular mesh again
    mesh = o3d.io.read_triangle_mesh(mesh_filename)

    # Smooth the mesh with Taubin
    l = options.taubin_smooth
    print('filter with Taubin with ' + str(l) + ' iterations')
    mesh = mesh.filter_smooth_taubin(number_of_iterations=l)
    mesh = coordinate_correction(mesh, mesh_filename)

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
    
    
    ## Save the generated images on the target object
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
    result_dict["nerf_scale"] = options.nerf_scale

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

    # Translation part
    d = np.array([[0], [dist], [0]])

    # Transformation matrix at the initial pose
    trans_initial = np.vstack((np.hstack((rot, d)), np.array([0, 0, 0, 1])))

    # Number of samples vertically
    h_num = 8
    # Number of samples along the circle around the object
    i_num = 12
    angle_dev_i = 2*np.pi/i_num
    angle_dev_h = (np.pi/2)/h_num

    # Initialize all the camera view pose candidates & take the picture
    for h in range(h_num): # Rotate around axis-x
        rot_angle_h = h * angle_dev_h
        # Rotation matrix for the camera pose around x-axis
        rot_around_x = R.from_quat([np.sin(rot_angle_h/2), 0, 0, np.cos(rot_angle_h/2)]).as_matrix()
        camera_pose_vet = np.matmul(
                np.vstack(
                    (np.hstack((rot_around_x, np.array([[0], [0], [0]]))), 
                    np.array([0, 0, 0, 1]))),
                trans_initial
                )
        for i in range(i_num): # Rotate around axis-z
            rot_angle_i  = i * angle_dev_i

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
            camera_frame["file_path"] = "./images/" + str(h) + "_" + str(i) + ".png"
            camera_frame["transform_matrix"] = camera_pose.tolist()
            result_dict["frames"].append(camera_frame)

            # Setting current transformation matrix.
            testbed.set_nerf_camera_matrix(camera_pose[:-1])
            # Formally take a picture at that pose
            frame = testbed.render(resolution[0],resolution[1],8,linear=False)

            # Save the image to the folder
            plt.imsave(images_folder + "/" + str(h) + "_" + str(i) + ".png", \
                       frame.copy(order='C'))

    # Dump the dict to the json file
    json.dump(result_dict, json_file, indent=4)


    return normalize_stats, csv_filename

if __name__ == "__main__":
    """Command line interface."""
    parser = argparse.ArgumentParser()

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
        '--taubin-smooth', '-l', type=int, default=50,
        help='Specify the Taubin smooth iterations to reduce noise'
    )
    parser.add_argument(
        '--train', action = 'store_true', help="Re-split the mesh into primitives"
    )
    parser.add_argument(
        '--store', action = 'store_true', help="Store the new splitting results"
    )

    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true', help="Whether to activate the visualization platform"
    )
    parser.add_argument(
        '--depth', action = 'store_true', help="Whether to collect the depth data as well"
    )
    add_mp_parameters(parser)
    parser.set_defaults(normalize=False)
    parser.set_defaults(train=False)
    parser.set_defaults(store=True)
    parser.set_defaults(visualization=False)

    # Arguments for pictures
    parser.add_argument('--distance', \
                        type=float, default=1.5, \
                        help="Approximate Distance between the center of the \
                            target object & the robot base")
    parser.add_argument('--nerf-scale', \
                        type=float, default=1, \
                        help="The value used to scale the real scene into the\
                            scene contained in an unit cube used by instant-NGP")
    options = parser.parse_args(sys.argv[1:])
    camera_intrinsics_dict = {}
    ## NOTE: Attributes for the phone camera
    # camera_intrinsics_dict["w"] = 1080
    # camera_intrinsics_dict["h"] = 1920
    
    # camera_intrinsics_dict["fl_x"] = 1433.7435988377988,
    # camera_intrinsics_dict["k1"] = 0.0
    # camera_intrinsics_dict["p1"] = 0.0
    # camera_intrinsics_dict["fl_y"] = 1438.1064292212557
    # camera_intrinsics_dict["k2"] = 0.0
    # camera_intrinsics_dict["p2"] = 0.0
    # camera_intrinsics_dict["cx"] = 540.1934198798218
    # camera_intrinsics_dict["cy"] = 955.7285435381777
    # camera_intrinsics_dict["camera_angle_x"] = 0.7204090661409585
    # camera_intrinsics_dict["camera_angle_y"] = 1.1772201404076283

    nerf_dataset = "./data/table1_real"
    ## NOTE: Attributes for the camera on the real robot
    camera_intrinsics_dict["w"] = 640
    camera_intrinsics_dict["h"] = 480
    
    camera_intrinsics_dict["fl_x"] = 552.0291012161067
    camera_intrinsics_dict["k1"] = 0.0
    camera_intrinsics_dict["p1"] = 0.0
    camera_intrinsics_dict["fl_y"] = 552.0291012161067
    camera_intrinsics_dict["k2"] = 0.0
    camera_intrinsics_dict["p2"] = 0.0
    camera_intrinsics_dict["cx"] = 320
    camera_intrinsics_dict["cy"] = 240
    camera_intrinsics_dict["camera_angle_x"] = 1.050688
    camera_intrinsics_dict["camera_angle_y"] = 0.8202161279220551
    # sdf_normalize_stats, csv_filename = preprocess(camera_intrinsics_dict, options.distance, nerf_dataset, \
    #            "base_upper.ingp", options)

    csv_filename = os.path.join(nerf_dataset, "target_obj.csv")
    ######
    ## Part I: embed pose estimation
    ######
    # Obtain the reference images & Test pose estimation
    img_file = "pose_estimation_masked_rgb.png"
    img_dir = nerf_dataset
    images_reference_list = \
        ["/images/" + x for x in \
         os.listdir(nerf_dataset + "/images")]# All images under foler "images"
    mesh_filename = nerf_dataset + "/target_obj.obj"
    mesh = o3d.io.read_triangle_mesh(mesh_filename)

    camera_pose_est, camera_proj_img, nerf_scale = \
                estimate_camera_pose("/" + img_file, img_dir, images_reference_list, \
                         mesh, None, \
                         image_type = 'outdoor', visualization = options.visualization)

    # camera_pose_est = np.array([
    #             [
    #                 -0.9659258262890682,
    #                 5.746937261686305e-17,
    #                 -0.25881904510252063,
    #                 -1.035276180410083
    #             ],
    #             [
    #                 -0.25881904510252063,
    #                 -2.1447861848524057e-16,
    #                 0.9659258262890682,
    #                 3.863703305156273
    #             ],
    #             [
    #                 0.0,
    #                 1.0,
    #                 2.220446049250313e-16,
    #                 0.5
    #             ],
    #             [
    #                 0.0,
    #                 0.0,
    #                 0.0,
    #                 1.0
    #             ]
    #         ])
    # nerf_scale = 2.5
    ## TODO: To test the performance of contact graspnet under different camera poses, 
    ## there is indeed a need to specify camera_pose_est at different poses
    # camera_pose_est = camera_pose_gt

    # a = R.from_euler('zyx', [0, 30, 0], degrees=True).as_matrix()
    # a = np.vstack((np.hstack((a, [[0], [0], [0]])), np.array([0, 0, 0, 1])))

    # # b = R.from_euler('zyx', [20, 0, 0], degrees=True).as_matrix()
    # # b = np.vstack((np.hstack((b, [[0], [0], [0]])), np.array([0, 0, 0, 1])))
    # camera_pose_est = a@camera_pose_est

    ## Debug: Take a picture at the estimated camera pose
    # Construct the instant-NGP testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    #Loading Snapshot & Figure out parameters
    testbed.load_snapshot(os.path.join(nerf_dataset, "base.ingp"))
    resolution = [camera_intrinsics_dict["w"], camera_intrinsics_dict["h"]]
    fov_x = camera_intrinsics_dict["camera_angle_x"] * 180/np.pi
    fov_y = camera_intrinsics_dict["camera_angle_y"] * 180/np.pi
    testbed.fov = max(fov_x, fov_y)

    
    testbed.render_mode = ngp.Shade
    # Take a picture at the estimated pose
    # Setting current transformation matrix.
    camera_pose_est = np.matmul(camera_pose_est, \
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
   
    testbed.set_nerf_camera_matrix(camera_pose_est[:-1])
    # Formally take a picture at that pose
    frame = testbed.render(resolution[0],resolution[1],8,linear=False)

    # Save the image to the folder
    plt.imsave("pose_estimation_debug.png", \
                frame.copy(order='C'))
    # Restore the frame convention (in open3d)
    camera_pose_est = np.matmul(camera_pose_est, \
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
   
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

    # Correct the frame conventions in camera & gripper of SPOT
    gripper_pose_current = camera_pose_est@np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    
    # Correction between the hand camera & the center of robot gripper
    gripper_z_local = (gripper_pose_current[:3, :3].T)[0:3, 2]

    # The gripper's z axis should be vertical
    angle = np.arccos(np.dot([0, 0, 1], gripper_z_local))
    gripper_z_local_cross = np.cross(np.array([0, 0, 1]), gripper_z_local)
    gripper_z_local_cross /= np.linalg.norm(gripper_z_local_cross)
    camera_gripper_correction = R.from_quat(\
        [gripper_z_local_cross[0] * np.sin(angle/2), \
         gripper_z_local_cross[1] * np.sin(angle/2), \
         gripper_z_local_cross[2] * np.sin(angle/2), np.cos(angle/2)]).as_matrix()
    camera_gripper_correction = np.vstack(\
        (np.hstack((camera_gripper_correction, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
    gripper_pose_current = gripper_pose_current@camera_gripper_correction

    # NOTE: Manually set up the height of the gripper
    # gripper_pose_current[2, 3] = 3.75*0.0254*nerf_scale
     # Correction between the hand camera & the center of robot gripper
    # camera_gripper_correction = R.from_quat(\
    #     [0, np.sin(np.pi/72), 0, np.cos(np.pi/72)]).as_matrix()
    # camera_gripper_correction = np.vstack(\
    #     (np.hstack((camera_gripper_correction, np.array([[0], [0], [-0.0254]]))), np.array([0, 0, 0, 1])))
    # gripper_pose_current = gripper_pose_current@camera_gripper_correction
    
    method = "sq_split" 
    if method == "sq_split":
        grasp_poses_world = predict_grasp_pose_sq(gripper_pose_current, \
                            mesh, csv_filename, \
                            [0.0, 1.0], stored_stats_filename, \
                                gripper_attr, options)
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

            if method == "cg":
                # Read the vertices of the whole mesh, as well as the 
                # convert the scene into the real-world scale
                pc_full = np.array(mesh.vertices)
                pc_full /= nerf_scale
                camera_extrinsics[0:3, 3] /= nerf_scale
            

                pc_full = np.linalg.inv(camera_extrinsics)@(np.vstack((pc_full.T, np.ones(pc_full.shape[0]))))
                pc_full = (pc_full[0:3, :]).T
                
                pc_colors = np.array(mesh.vertex_colors)
                # Predict grasp poses based on the whole mesh
                grasp_poses_cg = predict_grasp_pose_contact_graspnet(\
                            pc_full, camera_intrinsics_matrix, pc_colors=pc_colors,\
                            filter_grasps=False, local_regions=False,\
                            mode = "xyz", visualization=False)
            elif method == "cg_depth":
                # Obtain the depth array in instant-NGP's scale
                depth_array = depth_map_mesh(mesh, camera_intrinsics_matrix, camera_extrinsics)
                print(depth_array.shape)
                # Convert the value into the real-world scene
                depth_array /= nerf_scale
                camera_extrinsics[0:3, 3] /= nerf_scale

                # Save the depth array for debugging purpose
                depth_array_save = depth_array * (65536/2)
                Image.fromarray(depth_array_save.astype('uint16')).save("./depth_test.png")
                # Predict grasp poses on the Depth array (in real-world scale)
                grasp_poses_cg = predict_grasp_pose_contact_graspnet(\
                            depth_array, camera_intrinsics_matrix, pc_colors=None,\
                            filter_grasps=False, local_regions=False,\
                            mode = "depth", visualization=True)
            
    
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
                            camera_extrinsics, visualization = options.visualization)

    tran_norm_min = np.Inf
    grasp_pose_gripper = np.eye(4)
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
        
        # Consider the symmetry along x-axis
        r = R.from_matrix(tran[:3, :3])
        r_xyz = r.as_euler('xyz', degrees=True)

        # Rotation along x-axis is symmetric
        if r_xyz[0] > 90:
            r_xyz[0] = r_xyz[0] - 180
        elif r_xyz[0] < -90:
            r_xyz[0] = r_xyz[0] + 180
        tran = R.from_euler('xyz', r_xyz, degrees=True).as_matrix()
        # Find the one with the minimum "distance"
        tran_norm = np.linalg.norm(tran - np.eye(3))

        if tran_norm < tran_norm_min:
            tran_norm_min = tran_norm
            grasp_pose_gripper = grasp_pose

    rel_transform_gripper = np.linalg.inv(gripper_pose_current)@grasp_pose_gripper
    # Consider the symmetry along x-axis
    r = R.from_matrix(rel_transform_gripper[:3, :3])
    r_xyz = r.as_euler('xyz', degrees=True)
    print("==Test==")
    print(r_xyz)
    # Rotation along x-axis is symmetric
    if r_xyz[0] > 90:
        r_xyz[0] = r_xyz[0] - 180
    elif r_xyz[0] < -90:
        r_xyz[0] = r_xyz[0] + 180
    rel_transform_gripper[:3, :3]= R.from_euler('xyz', r_xyz, degrees=True).as_matrix()
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
    # Command the robot to grasp
    print(gripper_pose_current)
    print(grasp_pose_gripper)

    print(rel_transform_gripper)
    print(r_xyz)
