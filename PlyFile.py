import copy
from libs.plyfile import PlyData
from config import *
import numpy as np
from utils import compute_rotation_to_align_vectors
from config import Config as cfg

class PlyFile(object):
    #: Header for exporting point cloud to PLY
    ply_header = (
        '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {face_count}
property list uchar int vertex_indices
end_header
''')

    def __init__(self, coordinates, color=[255, 0, 0], faces=None):
        self.coordinates = coordinates
        if isinstance(color, list):
            self.colors = np.ones(coordinates.shape) * color
        elif isinstance(color, np.ndarray):
            self.colors = color
        self.faces = faces

    def __str__(self):
        if self.faces is None:
            return "coordinates " + str(self.coordinates.shape) + " - faces " + str(0)
        else:
            return "coordinates " + str(self.coordinates.shape) + " - faces " + str(self.faces.shape)

    def append_points(self, vertices, color, faces=None):
        if faces is not None:
            faces[:, 1:4] = faces[:, 1:4] + self.coordinates.shape[0]
            if self.faces is None:
                self.faces = faces
            else:
                self.faces = np.vstack((self.faces, faces))
        self.coordinates = np.vstack((self.coordinates, vertices))
        colors = np.ones(vertices.shape) * color
        self.colors = np.vstack((self.colors, colors))

    def write_ply(self, output_file):
        """
        Generate a ply file from the attributes of the class
        :param output_file:
        :return:
        """

        base_dir = os.path.dirname(output_file)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        points = np.hstack([self.coordinates, self.colors])
        try:
            if self.faces is not None:
                with open(output_file, 'w') as outfile:
                    outfile.write(self.ply_header.format(
                        vertex_count=len(points), face_count=len(self.faces)))
                    np.savetxt(outfile, points, '%f %f %f %d %d %d')
                    np.savetxt(outfile, self.faces, '%u %u %u %u')
                    print("Point Cloud file generated!")
            else:
                with open(output_file, 'w') as outfile:
                    outfile.write(self.ply_header.format(
                        vertex_count=len(points), face_count=0))
                    np.savetxt(outfile, points, '%f %f %f %d %d %d')
                    print("Point Cloud file generated!")
        except Exception as e:
            raise Exception("Could not generate pointcloud")

    @staticmethod
    def read_ply(input_file):
        """
        Read a ply file and return the vertices and faces as numpy vectors
        :param input_file:
        :return:
        """

        plydata = PlyData.read(input_file)

        vertices = [[vertex['x'], vertex['y'],vertex['z']] for vertex in plydata['vertex'].data]
        vertices_np = np.asarray(vertices)

        faces = plydata['face'].data['vertex_indices']
        faces = np.asarray([face for face in faces])

        threes = np.ones((faces.shape[0], 1), dtype=np.uint32) * 3
        faces_np = np.hstack((threes, faces))

        return vertices_np, faces_np


    def draw_arrows_for_sceneflow(self, vertices, sceneflow, arrow_vertices, arrow_faces, color=[255,0,0]):
        """
        :param vertices:
        :param sceneflow:
        :param arrow_vertices:
        :param arrow_faces:
        :param color:
        :return:
        """

        ## Go through all vertices and
        ## transform arrows to point in direction of motion vectors
        for i in range(len(vertices)):
            ## Make a copy of the arrow mesh features for every vertex
            arrow_v = copy.deepcopy(arrow_vertices)
            arrow_f = copy.deepcopy(arrow_faces)

            vertex = vertices[i]
            sceneflow_vertex = sceneflow[i]
            sceneflow_norm = np.linalg.norm(sceneflow_vertex)
            sceneflow_norm_as_vector = np.zeros_like(vertex)
            sceneflow_norm_as_vector[2] = sceneflow_norm

            ## Before transforming the arrow, scale the arrow in its longitudinal
            ## dimension according to the sceneflow norm
            arrow_v_old = copy.deepcopy(arrow_v)
            arrow_v[np.where(arrow_v[:,2] != 0)] = arrow_v[np.where(arrow_v[:,2] != 0)] + sceneflow_norm_as_vector

            ## Now scale the arrow to be a percentage of the voxel grid size
            arrow_v = arrow_v * (0.01 * cfg.n_voxels)

            R = compute_rotation_to_align_vectors(sceneflow_vertex)
            ## Rotate points of arrow mesh according to the current sceneflow vector
            arrow_v = arrow_v.dot(R.transpose())
            ## Translate points of arrow mesh to the location of the current vertex
            arrow_v = arrow_v + vertex
            self.append_points(arrow_v, color, faces=arrow_f)