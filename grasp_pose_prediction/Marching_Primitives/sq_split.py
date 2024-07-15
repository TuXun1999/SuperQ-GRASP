import numpy as np
import sys
import open3d as o3d
import csv
from Marching_Primitives.MPS import MPS, eul2rotm, parseInputArgs
import scipy.io
def create_superellipsoids(e1, e2, a1, a2, a3):
    pc_shape = 50
    pc = np.zeros((pc_shape * pc_shape, 3))

    eta = np.linspace(-np.pi/2, np.pi/2, pc_shape, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, pc_shape, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)

    t1 = np.sign(np.cos(eta)) * np.power(abs(np.cos(eta)), e1)
    t2 = np.sign(np.sin(eta)) * np.power(abs(np.sin(eta)), e1)
    t3 = np.sign(np.cos(omega)) * np.power(abs(np.cos(omega)), e2)
    t4 = np.sign(np.sin(omega)) * np.power(abs(np.sin(omega)), e2)
    pc[:, 0] = (a1 * t1 * t3).flatten()
    pc[:, 1] = (a2 * t1 * t4).flatten()
    pc[:, 2] = (a3 * t2).flatten()

    return pc
def sq_predict_mp(csvfile_path, args = None):
    '''
    The function to split the target object into several superquadrics as
    primitives, using the algorithm mentioned in "Marching Primitives"
    '''
    # Read the csv file & Extract out SDF
    sdf = []
    with open(csvfile_path, newline='') as csvfile:
        sdf_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in sdf_reader:
            sdf.append(float(row[0]))
    sdf = np.array(sdf)

    # Build up the voxel grid
    voxelGrid = {}
    voxelGrid['size'] = (np.ones(3) * sdf[0]).astype(int)
    voxelGrid['range'] = sdf[1:7]
    sdf = sdf[7:]

    voxelGrid['x'] = np.linspace(float(voxelGrid['range'][0]), float(voxelGrid['range'][1]), int(voxelGrid['size'][0]))
    voxelGrid['y'] = np.linspace(float(voxelGrid['range'][2]), float(voxelGrid['range'][3]), int(voxelGrid['size'][1]))
    voxelGrid['z'] = np.linspace(float(voxelGrid['range'][4]), float(voxelGrid['range'][5]), int(voxelGrid['size'][2]))
    x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'])

    # Permute the order (different data orders in Matlab & Python)
    # NOTE: This part is just trying to obey the data order convention in original 
    # Matlab program. There might be a way to continue the program even in Python data order
    x = np.transpose(x, (1, 0, 2))
    y = np.transpose(y, (1, 0, 2))
    z = np.transpose(z, (1, 0, 2))

    # Construct the points in voxelGrid 
    s = np.stack([x, y, z], axis=3)
    s = s.reshape(-1, 3, order='F').T

    # Construct the voxel grid
    voxelGrid['points'] = s

    voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
    voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
    voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
    voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

    # Complte extracting out the sdf
    sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])

    # marching-primitives
    import time

    # Parsing varargin
    if args is None:
        para = parseInputArgs(voxelGrid, sys.argv[1:])
    else:
        para = args
        # Add the values dependent on the voxelGrid
        para.minArea = np.ceil(voxelGrid["size"][0] / 20)
        para.paddingSize = np.ceil(12 * voxelGrid["truncation"] / voxelGrid["interval"])
        para.nanRange = 0.5 * voxelGrid["interval"]

        
    start_time = time.time()
    x = MPS(sdf, voxelGrid, para) 
    # This line is to read results from Matlab programs (mainly for debugging purpose)
    # x = scipy.io.loadmat('matlab_file.mat').get('x')
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return x

if __name__ == "__main__":
    # Loading file paths
    csvfile_path = "../data/chair6_normalized.csv" # Specify the location of the csv file
    csvfile_path_list = csvfile_path.split('.')

    # Verify that the selected file is in csv format
    if csvfile_path_list[-1] != "csv":
        print("Please select a csv file!!")

    mesh_filename = csvfile_path
    normalize_suffix_idx = mesh_filename.find("_normalized.csv")
    # Check whether the model is normalized
    if normalize_suffix_idx != -1:
        mesh_filename = mesh_filename[0:normalize_suffix_idx] + ".obj"
    else:
        csv_suffix_idx = mesh_filename.find(".csv")
        mesh_filename = mesh_filename[0:csv_suffix_idx] + ".obj"

    # Obtain the parameters of the predicted sq's
    x = sq_predict_mp(csvfile_path)
    # Read the original mesh file (open3d is used here for visualization)
    # TODO: could use other libraries, such as trimehs, maya, etc. 
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    if normalize_suffix_idx != -1:
        # Normalize the original mesh (for visualization purpose)
        mesh_scale = 0.8
        vertices = np.asarray(mesh.vertices)
        bbmin = np.min(vertices, axis=0)
        bbmax = np.max(vertices, axis=0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

    sq_mesh = []
    for i in range(x.shape[0]):
        # Read the superquadrics' parameters
        e1, e2, a1, a2, a3, r, p, y, t1, t2, t3 = x[i, :]
        if e1 < 0.01:
            e1 = 0.01
        if e2 < 0.01:
            e2 = 0.01

        # Custom function to sample points on the superquadrics
        sq_vertices = create_superellipsoids(e1, e2, a1, a2, a3)
        rot = eul2rotm(np.array([r, p, y]))
        sq_vertices = np.matmul(rot, sq_vertices.T).T + np.array([t1, t2, t3])

        # Construct a point cloud representing the reconstructed object mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sq_vertices)
        # Visualize the super-ellipsoids
        pcd.paint_uniform_color((0.0, 0.4, 0.0))

        sq_mesh.append(pcd)
    # Create the window to display everything
    vis= o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    for val in sq_mesh:
        vis.add_geometry(val)
    vis.run()

    # Close all windows
    vis.destroy_window()
'''
TODO:
This is the part to save the mesh into the format of mat & stl
'''
# stl = mesh.to_stl()

# ifsave = True
# pathname, name, ext = os.path.splitext(os.path.join(file_path, file))
# if ifsave:
#     savemat(os.path.join(pathname, f"{name}_sq.mat"), {'x_save': x.astype(np.float32)})
#     stl.save(os.path.join(pathname, f"{name}_sq.stl"), mode=stl.Mode.BINARY)

