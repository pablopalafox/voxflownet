from libs.plyfile import PlyData
from utils_vg import PlyFile
import numpy as np

## Cone
cone_plydata = PlyData.read('ply_examples/cone.ply')
cone = cone_plydata['vertex'].data
cone_np = cone.view(np.float32).reshape(cone.shape[0], 3)
cone_faces = cone_plydata['face'].data['vertex_indices']
faces = np.asarray([face for face in cone_faces])
threes = np.ones((faces.shape[0], 1), dtype=np.uint32) * 3
cone_faces_np = np.hstack((threes, faces))

## Cylinder
cylinder_plydata = PlyData.read('ply_examples/cylinder.ply')
cylinder = cylinder_plydata['vertex'].data
cylinder_np = cylinder.view(np.float32).reshape(cylinder.shape[0], 3)
cylinder_faces = cylinder_plydata['face'].data['vertex_indices']
faces = np.asarray([face for face in cylinder_faces]) + cone_np.shape[0]
threes = np.ones((faces.shape[0], 1), dtype=np.uint32) * 3
cylinder_faces_np = np.hstack((threes, faces))

## Make cylinder smaller
cylinder_np[:, 2] = cylinder_np[:, 2] * 4.0
cylinder_np[:, 0] = cylinder_np[:, 0] / 2.0
cylinder_np[:, 1] = cylinder_np[:, 1] / 2.0
cylinder_np /= 2.0

## Transform cone
##############################################
Rx_90 = np.matrix('1 0 0; 0 0 -1; 0 1 0')
Rx_minus90 = np.matrix('1 0 0; 0 0 1; 0 -1 0')
Ry_90 = np.matrix('0 0 1; 0 1 0; -1 0 0')
Ry_180 = np.matrix('-1 0 0; 0 1 0; 0 0 -1')
Rz = np.matrix('0 -1 0; 1 0 0; 0 0 1')
##############################################
cone_np = cone_np - 1.0
cone_np = cone_np.dot(Rx_minus90.transpose())
cone_np = cone_np.dot(Ry_180.transpose())
cone_np[:, 2] += 4
## compute centroid of cone
cone_centroid = np.mean(cone_np, axis=0)
print(cone_centroid)
cone_np[:,0] = cone_np[:,0] - cone_centroid[0,0]
cone_np[:,1] = cone_np[:,1] - cone_centroid[0,1]


## Join
vertices = np.vstack((cone_np, cylinder_np))
vertices = vertices / vertices.max()
vertices[:,2] = vertices[:,2] + 0.52
faces = np.vstack((cone_faces_np, cylinder_faces_np))

plyfile = PlyFile(vertices, color=[255, 0, 0], faces=faces)
plyfile.write_ply("ply_examples/awesome_arrow.ply")