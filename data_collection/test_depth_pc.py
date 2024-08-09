import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
# Read the mesh & Display the interactive window
mesh = o3d.io.read_triangle_mesh("../data/chair2_real/target_obj.obj")
mesh_tri = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(mesh_tri)

# Specify the camera
camera_pose_gt = np.array([
        [
          0.3288268137685253,
          -0.47246782519371655,
          0.8177084326161024,
          3.4995952980181464
        ],
        [
          0.9436957092008466,
          0.13118628743264144,
          -0.3036915646276309,
          -1.5575937599487182
        ],
        [
          0.0362123595920231,
          0.8715298688020476,
          0.48900342827736343,
          1.9588267493064477
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])
camera_pose_gt = np.matmul(camera_pose_gt, \
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

a = R.from_euler('zyx', [0, -90, 0], degrees=True).as_matrix()
a = np.vstack((np.hstack((a, [[0], [0], [0]])), np.array([0, 0, 0, 1])))




camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
camera_frame.scale(10/64, [0, 0, 0])
camera_frame.transform(camera_pose_gt)
fl_x = 552.0291012161067
fl_y = 552.0291012161067
cx = 320
cy = 240
camera_intrinsics_matrix = np.array([
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1]
            ])
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_matrix = o3d.core.Tensor(camera_intrinsics_matrix),
    extrinsic_matrix = o3d.core.Tensor(np.linalg.inv(camera_pose_gt)),
    width_px=640,
    height_px=480,
)
# We can directly pass the rays tensor to the cast_rays function.
ans = scene.cast_rays(rays)

print(ans['t_hit'].shape)
hit = ans['t_hit'].isfinite()
points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
points = points.numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color((1, 0, 0))



# Visualization
# Visualize cloud and edit
vis = o3d.visualization.Visualizer()

height=1080
width=1920
vis.create_window(height=height, width=width)



#vis.add_geometry(pcd)
vis.add_geometry(mesh)
vis.add_geometry(camera_frame)

ctr = vis.get_view_control()
x = -100
y = -350
ctr.rotate(x, y, xo=0.0, yo=0.0)
ctr.translate(0, 0, xo=0.0, yo=0.0)
ctr.scale(0.01)
# Updates
# vis.update_geometry(pcd)
# vis.update_geometry(mesh)
# vis.update_geometry(camera_frame)
vis.poll_events()
vis.update_renderer()

# Capture image

vis.capture_screen_image('cameraparams.png')
vis.run()


vis.destroy_window()