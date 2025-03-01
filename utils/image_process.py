import json
import numpy as np
import cv2
import copy
import os
from scipy.optimize import fsolve
import torch
from LoFTR.src.utils.plotting import make_matching_figure
import matplotlib.cm as cm
from utils.mesh_process import point_select_in_space
##########################
## Part I: Manually Selected Point
##########################
# mouse callback function
g_image_click = None
g_image_display = None
def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = copy.deepcopy(g_image_display)
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    elif not (g_image_click is None):
        # Draw some lines on the image.
        # to indicate the location of the selected point
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, g_image_click[1]), (width, g_image_click[1]), color, thickness)
        cv2.line(clone, (g_image_click[0], 0), (g_image_click[0], height), color, thickness)
        cv2.circle(clone, g_image_click, radius = 4, color=color)
        cv2.imshow(image_title, clone)
    else:
        # Draw some lines on the imaege.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def get_pick_vec_manual_force(img):
    global g_image_display, g_image_click
    g_image_display = img

    # Show the image to the user and wait for them to click on a pixel
    image_title = 'Click'
    cv2.namedWindow(image_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    print("==================")
    print("Show the image for the user to click")
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # The user decides not to pick anything in the frame
            return 0.0, 0.0
    return g_image_click[0], g_image_click[1]
def read_proj_from_json(dataset_dir, img_name):
    '''
    Function to read the properties of that image directly from json file
    Return:
    proj: projection matrix of camera
    camera_pose: the pose of the camera
    NOTE: TO facilitate the following calculations in pinhole model and raycasting distance,
    in such case, the z-direction of the returned camera pose will point
    TOWARDS the object, AWAY from the camera plane.
    The same as the convention in https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

    nerf_scale: scale that downsamples the actual scene into the normalized box in nerf
    '''
    # Read the json file
    json_filename = dataset_dir + "/transforms.json"
    f = open(json_filename)
    data = json.load(f)

    # Read the scale that converts the real-world scene into the unit cube, if it exists
    # (this is to enhance the performance)
    if data.get("nerf_scale") is not None:
        nerf_scale = data["nerf_scale"]
    else:
        nerf_scale = 1

    # Read the camera attributes
    fl_x = data["fl_x"]
    fl_y = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]

    # TODO: incorporate other distortion coefficients into concern as well
    proj = np.array([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0]
    ])
    proj = np.float32(proj) # Converted into float type
    # By default, it's assumed that the camera is at the origin
    camera_pose = np.eye(4)
    # Search for the frame with the name matched
    for frame in data["frames"]:
        if frame["file_path"] == "." + img_name: 
            # When recording the name of the image, "." is always included
            camera_pose = frame["transform_matrix"]
            camera_pose = np.array(camera_pose)
            break
    # Conversion between the usual camera frame (in Pyrender) & 
    # the frame convention in instant-NGP
    # (Please check out the README file for more details)
    camera_pose = np.matmul(\
        camera_pose, \
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0 , 1]]))

    return proj, camera_pose, nerf_scale


def point_select_from_image(img_dir, img_file, save_fig = False):
    '''
    The function to select one point on one training image 
    '''
    ## Part I: open an image
    # Open the image (human choice)
    image = cv2.imread(os.path.join(img_dir, img_file))

    # Select a point manually
    pick_x, pick_y = get_pick_vec_manual_force(image)
    print("================")
    print("Clicked Point")
    print(pick_x)
    print(pick_y)
    if save_fig:
        # Save the image 
        cv2.circle(image, (pick_x, pick_y), radius = 8, color=(0, 0, 255), thickness=-1)
        cv2.imwrite("image_plus_point_selected.jpg", image)
    ## Part II: obtain the corresponding camera projection matrix
    camera_proj, camera_pose,nerf_scale = read_proj_from_json(img_dir, img_file)
    print("Camera pose in world frame: ")
    print(camera_pose)


    ## Part III: solve out the depth ambiguity (skipped)

    ## Part IV: find the ray direction in camera frame
    initial_guess = [1, 1, 10]
    def equations(vars):
        x, y, z = vars
        eq = [
            camera_proj[0][0] * x + camera_proj[0][1] * y + camera_proj[0][2] * z - pick_x * z,
            camera_proj[1][0] * x + camera_proj[1][1] * y + camera_proj[1][2] * z - pick_y * z,
            x * x + y * y + z * z - 10 * 10
        ]
        return eq

    root = fsolve(equations, initial_guess)
    ## Part V: convert the point coorindate in world frame
    dir_world = np.matmul(camera_pose, np.array([[root[0]], [root[1]], [root[2]], [0]]))
    cv2.destroyAllWindows()
    return dir_world[0:3, 0], camera_pose, camera_proj, nerf_scale

def point_select_from_custom_image(image_name, camera_instrinsics_proj, camera_pose_est, mesh):
    '''
    The function to select a point on the custom image
    '''
    # Select a point on the captured image at the specified pose
    image = cv2.imread(image_name)
    pick_x, pick_y = get_pick_vec_manual_force(image)
    print("================")
    print("Clicked Point")
    print(pick_x)
    print(pick_y)

    # Read the camera attributes
    camera_proj = camera_instrinsics_proj
    def equations(vars):
        x, y = vars
        eq = [
            camera_proj[0][0] * x + camera_proj[0][1] * y + camera_proj[0][2] * 1 - pick_x * 1,
            camera_proj[1][0] * x + camera_proj[1][1] * y + camera_proj[1][2] * 1 - pick_y * 1,
        ]
        return eq

    root = fsolve(equations, [0, 0])
    # Convert the point coorindate in world frame
    ray_dir = np.matmul(camera_pose_est, np.array([[root[0]], [root[1]], [1], [0]]))
   
    cv2.destroyAllWindows()

    # Normalize the directional vector

    ray_dir = ray_dir[0:3, 0]
    n = np.linalg.norm(ray_dir)
    ray_dir = ray_dir / n

    # Find the coordinate of the selected point in space using the 
    # raycasting functionality in open3d
    pos, dist  = point_select_in_space(camera_pose_est, ray_dir, mesh)

    return pos, dist


def dir_point_on_image(img_dir, img_file, pixel_coords):
    '''
    The function to return the directions of casting rays that pass through 
    the selected pixels on the image
    '''
    ## Part I: open an image
    # Open the image (human choice)
    image = cv2.imread(os.path.join(img_dir,img_file))
    
    ## Part II: obtain the corresponding camera projection matrix
    camera_proj, camera_pose,nerf_scale = read_proj_from_json(img_dir, img_file)
    print("Camera pose in world frame: ")
    print(camera_pose)


    ## Part III: solve out the depth ambiguity (skipped)

    ## Part IV: find the ray direction in camera frame
    # fsolve is not used here, since it might be too slow
    # Here, we need to solve out the equation:
    # camera_proj * [[x], [y], [z]] = [[px], [py]]
    # Since we only need the direction, we can force z to be 1 
    # Why 1? => same direction as we shoot the ray from the camera; avoid the other 
    # invalid solution
    a, b, c = camera_proj[0, 0:3]
    d, e, f = camera_proj[1, 0:3]
    M = np.array([[e, -b], [-d, a]])/(a*e-b*d)
    N = np.array([[b*f-c*e], [c*d-a*f]])/(a*e-b*d)
    points_xy = np.matmul(M, pixel_coords.T) + N
    points_camera = np.vstack((points_xy, np.ones(points_xy.shape[1])))
    ## Part V: Normalize the directional vectors & Convert them into world frame
    dir_norm = np.linalg.norm(points_camera, axis=0)
    dir_camera = points_camera / dir_norm
    dir_world = np.matmul(camera_pose, \
                np.vstack((dir_camera, np.zeros(dir_camera.shape[1]))))
    return (dir_world[0:3, :]).T, camera_pose, camera_proj, nerf_scale


################################
## Part II:  Matched Features between two images
################################
def match_coords(image0_name, image1_name, matcher, th = 0.4, \
                 scale_restore = False, save_fig = False):
    '''
    The function to return the pixel coordates of matched points on two images
    Input: 
        names of the two images; 
        th: threshold for matched points
        scale_restore: whether the project the coordinates back to the shape of
        original images
    Output: pixel coordinates of matched points
    '''
    # Read the images & Preprocessing
    img0_raw = cv2.imread(image0_name, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)
    if scale_restore == True:
        h0 = img0_raw.shape[0]
        w0 = img0_raw.shape[1]
        h1 = img1_raw.shape[0]
        w1 = img1_raw.shape[1]
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    idx = mconf > th

    if save_fig: # Optional: save the results
        # Draw 
        color = cm.jet(mconf, alpha=0.7)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text,\
                            path="LoFTR-demo.jpg")
    if scale_restore:
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]
        mkpts0[:, 0] = mkpts0[:, 0] * (w0/640)
        mkpts0[:, 1] = mkpts0[:, 1] * (h0/480)
        mkpts1[:, 0] = mkpts1[:, 0] * (w1/640)
        mkpts1[:, 1] = mkpts1[:, 1] * (h1/480)
        return mkpts0, mkpts1
    else:
        return mkpts0[idx], mkpts1[idx]

def image_select_closest(image_name, images_reference_list, dataset_dir, matcher):
    '''
    The function to find the closest reference image to the sample image,
    the camera pose of which is regarded as the ground truth, and also will
    be used to calculate the pose of the sample image
    Input: image_name: the name of sample image, string
    images_reference_dir: the directory containing all reference images, List of Strings
    Output: 
    image_name_closest: the closest image name
    
    '''
    max_matched = 0
    image_name_closest = None
    for image_name_reference in images_reference_list:
        # For each reference image, check the number of matched points
        mpts0, _ = match_coords(dataset_dir + image_name, \
                                dataset_dir + image_name_reference, \
                                matcher)
        if mpts0.shape[0] > max_matched:
            max_matched = mpts0.shape[0]
            image_name_closest = image_name_reference
    _, camera_pose, nerf_scale = read_proj_from_json(dataset_dir, image_name_closest)
    return image_name_closest
        

    
