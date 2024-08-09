import open3d as o3d
import numpy as np
# Read the mesh & Display the interactive window
a = o3d.io.read_triangle_mesh("../data/chair2_real/target_obj.obj")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(a.vertices))
pcd.colors = o3d.utility.Vector3dVector(np.array(a.vertex_colors))
# Visualize cloud and edit
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()

vis.destroy_window()
i, j = vis.get_picked_points()[0], vis.get_picked_points()[1]

print(vis.get_picked_points()) 
dist = np.array(pcd.points[i]- pcd.points[j])
print(np.linalg.norm(dist))