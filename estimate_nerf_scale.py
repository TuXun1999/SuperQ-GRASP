import argparse
from argparse import Namespace
import sys
import os
import time
from scipy.optimize import fsolve

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import frame_helpers
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
import bosdyn.api.basic_command_pb2 as basic_command_pb2
from bosdyn.client import math_helpers
from bosdyn.api import \
    geometry_pb2
from bosdyn.client.robot_command import \
    RobotCommandBuilder, RobotCommandClient, \
        block_until_arm_arrives, blocking_stand, blocking_selfright
from bosdyn.util import seconds_to_duration
from bosdyn.client.math_helpers import SE3Pose, Quat

import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
import numpy as np
import json 

from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamPredictor
import supervision as sv
import torch
from torchvision.ops import box_convert

import open3d as o3d

from pose_estimation.pose_estimation import estimate_camera_pose
from grasp_pose_prediction.grasp_sq_mp import predict_grasp_pose_sq
from grasp_pose_prediction.grasp_contact_graspnet import \
    predict_grasp_pose_contact_graspnet, grasp_pose_eval_gripper_cg
from preprocess import instant_NGP_screenshot
'''
Helper function to determine the bounding box
'''
def bounding_box_predict(image_name, target, visualization=False):
    ## Predict the bounding box in the current image on the target object
    # Specify the paths to the model
    home_addr = os.path.join(os.getcwd(), "GroundingDINO")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG_PATH = home_addr + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = home_addr + "/weights/groundingdino_swint_ogc.pth"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH, DEVICE)

    IMAGE_PATH = image_name
    TEXT_PROMPT = target
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.3

    # Load the image & Do the bounding box prediction
    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    # Display the annotated frame
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    if visualization:
        sv.plot_image(annotated_frame, (16, 16))

    # Return the bounding box coordinates
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    if boxes.shape[0] == 0: # If there is no prediction
        print("Failed to Detect the target object in current image")
        print("Exiting...")
        return None
    elif boxes.shape[0] != 1: # If there are multiple target objects
        xyxy = xyxy[int(torch.argmax(logits))] 
        # Select the one with the highest confidence  
    else:
        xyxy = xyxy[0]
    return xyxy, logits[0]

def get_bounding_box_image(image, bbox, confidence):
    ## Impose the image with the bounding box also
    # Draw bounding boxes in the image
    polygon = []
    polygon.append([bbox[0], bbox[1]])
    polygon.append([bbox[2], bbox[1]])
    polygon.append([bbox[2], bbox[3]])
    polygon.append([bbox[0], bbox[3]])

    polygon = np.array(polygon, np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.polylines(image, [polygon], True, (0, 255, 0), 2)

    caption = "{} {:.3f}".format("Detected Bounding Box", confidence)
    cv2.putText(image, caption, (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    ##The function to predict the segmentation mask of the target object
    # using GroundingSAM
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_mask_image(image_source, xyxy):
    ## The function to generate an image with the background filtered out
    # Construct the SAM predictor
    home_addr = os.path.join(os.getcwd(), "GroundingDINO")
    SAM_CHECKPOINT_PATH = home_addr + "/sam_weights/sam_vit_h_4b8939.pth"
    SAM_ENCODER_VERSION = "vit_h"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # Predict the segmentation mask
    mask = segment(
        sam_predictor=sam_predictor,
        image=image_source,
        xyxy=xyxy.reshape(1, -1) # Unsqueeze the bbox coordinates
    )

    # Manually filter out the mask
    image_source = np.array(image_source)
    h, w, _ = image_source.shape
    image_masked = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if mask[0][i][j]:
                image_masked[i, j] = np.array(\
                    [image_source[i, j, 0], image_source[i, j, 1],\
                    image_source[i, j, 2], 255])
       
    return image_masked, mask


def estimate_obj_pose_hand(bbox, image_response, distance):
    ## Estimate the target object pose (indicated by the bounding box) in hand frame
    bbox_center = [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)]
    pick_x, pick_y = bbox_center
    # Obtain the camera inforamtion
    camera_info = image_response.source
    
    w = camera_info.cols
    h = camera_info.rows
    fl_x = camera_info.pinhole.intrinsics.focal_length.x
    k1= camera_info.pinhole.intrinsics.skew.x
    cx = camera_info.pinhole.intrinsics.principal_point.x
    fl_y = camera_info.pinhole.intrinsics.focal_length.y
    k2 = camera_info.pinhole.intrinsics.skew.y
    cy = camera_info.pinhole.intrinsics.principal_point.y

    pinhole_camera_proj = np.array([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0]
    ])
    pinhole_camera_proj = np.float32(pinhole_camera_proj) # Converted into float type
    # Calculate the object's pose in hand camera frame
    initial_guess = [1, 1, 10]
    def equations(vars):
        x, y, z = vars
        eq = [
            pinhole_camera_proj[0][0] * x + pinhole_camera_proj[0][1] * y + pinhole_camera_proj[0][2] * z - pick_x * z,
            pinhole_camera_proj[1][0] * x + pinhole_camera_proj[1][1] * y + pinhole_camera_proj[1][2] * z - pick_y * z,
            x * x + y * y + z * z - distance * distance
        ]
        return eq

    root = fsolve(equations, initial_guess)
    # Correct the frame conventions in hand frame & pinhole model
    # pinhole model: z-> towards object, x-> rightward, y-> downward
    # hand frame in SPOT: x-> towards object, y->rightward
    result = SE3Pose(x=root[2], y=-root[0], z=-root[1], rot=Quat(w=1, x=0, y=0, z=0))
    return result

'''
Helper functions to move the robot
'''
def compute_stand_location_and_yaw(vision_tform_target, robot_state_client,
                                distance_margin):

    # Compute drop-off location:
    #   Draw a line from Spot to the person
    #   Back up 2.0 meters on that line
    vision_tform_robot = frame_helpers.get_a_tform_b(
        robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)


    # Compute vector between robot and person
    robot_rt_person_ewrt_vision = [
        vision_tform_robot.x - vision_tform_target.x,
        vision_tform_robot.y - vision_tform_target.y,
        vision_tform_robot.z - vision_tform_target.z
    ]


    # Compute the unit vector.
    if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
        robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
    else:
        robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
            robot_rt_person_ewrt_vision)


    # Starting at the person, back up meters along the unit vector.
    drop_position_rt_vision = [
        vision_tform_target.x +
        robot_rt_person_ewrt_vision_hat[0] * distance_margin,
        vision_tform_target.y +
        robot_rt_person_ewrt_vision_hat[1] * distance_margin,
        vision_tform_target.z +
        robot_rt_person_ewrt_vision_hat[2] * distance_margin
    ]


    # We also want to compute a rotation (yaw) so that we will face the person when dropping.
    # We'll do this by computing a rotation matrix with X along
    #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
    xhat = -robot_rt_person_ewrt_vision_hat
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, xhat)
    mat = np.matrix([xhat, yhat, zhat]).transpose()
    heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

    return drop_position_rt_vision, heading_rt_vision

def get_walking_params(max_linear_vel, max_rotation_vel):
    max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                        angular=max_rotation_vel)
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params

def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=None, verbose=False):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded.
       Args:
        command_client: robot command client, used to request feedback
        cmd_id: command ID returned by the robot when the trajectory command was sent
        timeout_sec: optional number of seconds after which we'll return no matter what the
                        robot's state is.
        verbose: if we should print state at 10 Hz.
       Return values:
        True if reaches STATUS_AT_GOAL, False otherwise.
    """
    start_time = time.time()

    if timeout_sec is not None:
        end_time = start_time + timeout_sec
        now = time.time()

    while timeout_sec is None or now < end_time:
        feedback = command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the location')
            return

        time.sleep(0.1)
        now = time.time()

    if verbose:
        print('block_for_trajectory_cmd: timeout exceeded.')


def estimate_nerf_scale(options):
    if (options.image_source != "hand_color_image"):
        print("Currently Only Support Hand Camera!")
        return True
    ## Create an UI to take the user input & find the target object
    target = input("The target object you want to use for nerf scale estimataion - ")

    ## Fundamental Setup of the robotic platform
    bosdyn.client.util.setup_logging(options.verbose)
    # Authenticate with the robot
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    # Create the clients 
    image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    # Verification before the formal task
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    # Start of the formal task
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        ## Part 0: Start the robot and Power it on
        # Power on the robot
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. 
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        # Send the trajectory to the robot.
        cmd_id = command_client.robot_command(robot_command)

        time.sleep(0.5)
        

        ## Take one image & Estimate Pose
        image_responses = image_client.get_image_from_sources(\
            [options.image_source])
        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)
        image_name = os.path.join(options.nerf_model, "pose_estimation.jpg")
        cv2.imwrite(image_name, img)
        
        # Filter out the background
        xyxy, _ = bounding_box_predict(image_name, target)
        img, mask = get_mask_image(img, xyxy)

        # Save the new masked image (input for pose estimation module)
        data = Image.fromarray(img, 'RGBA') 
        data.save(os.path.join(options.nerf_model, "pose_estimation_masked.png")) 

        # Also save the rgb image for reference
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb, _ = get_mask_image(img_rgb, xyxy)

        data = Image.fromarray(img_rgb, 'RGBA')
        data.save(os.path.join(options.nerf_model, "pose_estimation_masked_rgb.png"))


        # Read the preprocessed data
        images_reference_list = \
            ["/images/" + x for x in \
            os.listdir(options.nerf_model + "/images")]# All images under foler "images"
        mesh_filename = os.path.join(options.nerf_model , "target_obj.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        camera_pose_est, _, _ = \
                estimate_camera_pose("/pose_estimation_masked.png", options.nerf_model, \
                                     images_reference_list, \
                         mesh, None, \
                         image_type = 'outdoor', visualization = False)
        
        ## Command the robot to extend its gripper for 0.5m ahead 
        gripper_dist = 0.5
        mat = np.array([
            [1, 0, 0, gripper_dist],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Build the SE(3) pose of the desired hand position in the moving body frame.
        hand_T_target = math_helpers.SE3Pose.from_matrix(mat)

        # Transform the desired pose from the moving body frame to the odom frame.
        robot_state = robot_state_client.get_robot_state()
        odom_T_hand = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.ODOM_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        
        odom_T_target = odom_T_hand * hand_T_target

        # duration in seconds
        seconds = 5.0

        # Create the arm command.
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_target.x, odom_T_target.y, odom_T_target.z, odom_T_target.rot.w, odom_T_target.rot.x,
            odom_T_target.rot.y, odom_T_target.rot.z, \
                frame_helpers.ODOM_FRAME_NAME, seconds)

        # Tell the robot's body to follow the arm
        follow_arm_command = RobotCommandBuilder.follow_arm_command()

        # Combine the arm and mobility commands into one synchronized command.
        command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

        # Send the request
        move_command_id = command_client.robot_command(command)
        print('Moving arm to position.')

        block_until_arm_arrives(command_client, move_command_id, 30.0)
        ## Take another image & Estimate Pose
        image_responses = image_client.get_image_from_sources(\
            [options.image_source])
        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)
        image_name = os.path.join(options.nerf_model, "pose_estimation.jpg")
        cv2.imwrite(image_name, img)
        
        # Filter out the background
        xyxy, _ = bounding_box_predict(image_name, target)
        img, mask = get_mask_image(img, xyxy)

        # Save the new masked image (input for pose estimation module)
        data = Image.fromarray(img, 'RGBA') 
        data.save(os.path.join(options.nerf_model, "pose_estimation_masked.png")) 

        # Also save the rgb image for reference
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb, _ = get_mask_image(img_rgb, xyxy)

        data = Image.fromarray(img_rgb, 'RGBA')
        data.save(os.path.join(options.nerf_model, "pose_estimation_masked_rgb.png"))


        camera_pose_est2, _, _ = \
                estimate_camera_pose("/pose_estimation_masked.png", options.nerf_model, \
                                     images_reference_list, \
                         mesh, None, \
                         image_type = 'outdoor', visualization = False)
        robot_state = robot_state_client.get_robot_state()
        odom_T_hand2 = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.ODOM_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        ## Figure out the nerf_scale and correct the json file
        dist = np.linalg.norm(camera_pose_est2[0:3, 3] - camera_pose_est[0:3, 3])
        gripper_dist = np.linalg.norm(odom_T_hand.get_translation()[0:3] \
                                      - odom_T_hand2.get_translation()[0:3])
        # In instant-NGP's world, the gripper has moved for "dist"
        nerf_scale = dist / gripper_dist

        print("========Estimated Nerf scale==========")
        print(nerf_scale)

        
        
        # If the user hopes to see the visualization
        if options.visualization:
            camera_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame()
            camera_frame_est.scale(10/64 * nerf_scale, [0, 0, 0])
            camera_frame_est.transform(camera_pose_est)
            camera_frame_est.paint_uniform_color((1, 0, 0))

            
            camera_frame2_est = o3d.geometry.TriangleMesh.create_coordinate_frame()
            camera_frame2_est.scale(10/64 * nerf_scale, [0, 0, 0])
            camera_frame2_est.transform(camera_pose_est2)
            camera_frame2_est.paint_uniform_color((1, 0, 0))
            # Create the window to display everything
            vis= o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            vis.add_geometry(camera_frame_est)
            vis.add_geometry(camera_frame2_est)

            vis.run()

            # Close all windows
            vis.destroy_window()

        input("Waiting for the user to stop")
        return True

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-source', help='Get image from source(s), \
                        only hand camera is supported so far', default='hand_color_image')
    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true', help="Whether to activate the visualization platform"
    )

    parser.add_argument('--nerf_model', help='The directory containing the preprocessed\
                         NeRF model to use')
    options = parser.parse_args(argv)
    try:
        estimate_nerf_scale(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False
if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)