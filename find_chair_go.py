import argparse
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
    RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_selfright
from bosdyn.util import seconds_to_duration
from bosdyn.client.math_helpers import SE3Pose, Quat

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate

import supervision as sv
import torch
from torchvision.ops import box_convert

from bosdyn.api import geometry_pb2 as geom
from bosdyn.api import world_object_pb2 as wo
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req
from bosdyn.util import now_timestamp

'''
Helper function to determine the bounding box
'''
def bounding_box_predict(image_name, target):
    ## Predict the bounding box in the current image on the target object
    # Specify the paths to the model
    home_addr = os.path.expanduser('~') + "/repo/multi-purpose-representation/GroundingDINO"

    CONFIG_PATH = home_addr + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = home_addr + "/weights/groundingdino_swint_ogc.pth"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    IMAGE_PATH = image_name
    TEXT_PROMPT = target
    BOX_TRESHOLD = 0.5
    TEXT_TRESHOLD = 0.5

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
    sv.plot_image(annotated_frame, (16, 16))

    # Return the bounding box coordinates
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if boxes.shape[0] != 1: # If there are multiple target objects
        try:
            xyxy = xyxy[int(torch.argmax(logits))] # Select the one with the highest confidence
        except:
            return None
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
    print("====Test Pose Estimation===")
    print(pinhole_camera_proj)
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
    print(root)
    # Correct the frame conventions in hand frame & pinhole model
    # pinhole model: z-> towards object, x-> rightward, y-> downward
    # hand frame in SPOT: x-> towards object, y->rightward
    result = SE3Pose(x=root[2], y=-root[0], z=-root[1], rot=Quat(w=1, x=0, y=0, z=0))
    print(result)
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

def find_chair_go(options):
    if (options.image_source != "hand_color_image"):
        print("Currently Only Support Hand Camera!")
        return True
    ## Create an UI to take the user input & find the target object
    target = input("What do you want to grasp?")

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
        ## Part 1: Take a picture & Determine the bounding box
        # Capture one image from the hand camera
        image_responses = image_client.get_image_from_sources(\
            [options.image_source])

        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)


        image_name = "./test.jpg"
        cv2.imwrite(image_name, img)
        bbox, confidence = bounding_box_predict(image_name, target)
        if bbox is None:
            robot.logger.info('Failed to Find any object close to your description')
            return True
        
        bbox_image = get_bounding_box_image(img, bbox, confidence)
        cv2.imwrite("./test_bbox.jpg", bbox_image)
        ## Estimate the orientation of the object & Approach it
        # Estimate the object's direction in hand frame
        hand_tform_obj = estimate_obj_pose_hand(bbox, image_responses[0],\
                                                options.distance)
        

        vision_tform_obj = frame_helpers.get_a_tform_b(
                        robot_state_client.get_robot_state(
                                        ).kinematic_state.transforms_snapshot,
                        frame_helpers.VISION_FRAME_NAME,
                        frame_helpers.HAND_FRAME_NAME) * hand_tform_obj
        print(vision_tform_obj)
        ## Command the robot to approach the target object
        # We now have found the target object
        drop_position_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
                vision_tform_obj, robot_state_client, distance_margin=1.2)
        
        print("====Before Movement===")
        print(drop_position_rt_vision)
        print(heading_rt_vision)
        # Tell the robot to go there
        # Limit the speed so we don't charge at the target object.
        move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=drop_position_rt_vision[0],
                goal_y=drop_position_rt_vision[1],
                goal_heading=heading_rt_vision,
                frame_name=frame_helpers.VISION_FRAME_NAME,
                params=get_walking_params(0.5, 0.5))
        
        end_time = 15.0
        cmd_id = command_client.robot_command(command=move_cmd,
                                                end_time_secs=time.time() +
                                                end_time)
       
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=15, verbose=True)

        ## Postlude: Let the user decide whether to further move the robot
        while True: 
            approach_res = input("Do you want to further move the robot? [Y/N]")
            if approach_res == 'N' or approach_res == 'no' or approach_res == 'n':
                break
            elif approach_res == 'Y' or approach_res == 'yes' or approach_res == 'y':
                # Ask how the user would like to move the robot
                move_option = input("Do you want to move closer or back to the original place to restart? [1/2]")
                if move_option == "1":
                    while True:
                        try:
                            more_dist = float(input("How much do you want to go further? [m]"))
                            break
                        except:
                            print("Please input a float value")
                    move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                        goal_x=more_dist,
                        goal_y=0,
                        goal_heading=0,
                        frame_name=frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME,
                        params=get_walking_params(0.5, 0.5))
            
                    end_time = 15.0
                    cmd_id = command_client.robot_command(command=move_cmd,
                                                            end_time_secs=time.time() +
                                                            end_time)
                
                    # Wait until the robot reports that it is at the goal.
                    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=15, verbose=True)
                elif move_option == "2":
                    print("Moving back to the original place; Restart the process with new parameters")
                    # Simply reverse the movement
                    move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                        goal_x=-drop_position_rt_vision[0],
                        goal_y=-drop_position_rt_vision[1],
                        goal_heading=-heading_rt_vision,
                        frame_name=frame_helpers.VISION_FRAME_NAME,
                        params=get_walking_params(0.5, 0.5))
            
                    end_time = 15.0
                    cmd_id = command_client.robot_command(command=move_cmd,
                                                            end_time_secs=time.time() +
                                                            end_time)
                
                    # Wait until the robot reports that it is at the goal.
                    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=15, verbose=True)
                else:
                    print("Could not understand your option. Exitting..")
                    return False
            else:
                print("Please input Y/N or yes/no or y/n")
        ## TODO: 
        # take_image_grasp(robot)
        # go_back()
    return True

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-source', help='Get image from source(s), \
                        only hand camera is supported so far', default='hand_color_image')
    parser.add_argument('--distance', \
                        type=float, default=1.5, \
                        help="Approximate Distance between the center of the \
                            target object & the robot base")
    options = parser.parse_args(argv)
    try:
        find_chair_go(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False
if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)