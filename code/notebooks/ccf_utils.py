"""CCF region lookup and mesh helpers used by the analysis notebooks.

`CCF` maps Common Coordinate Framework region ids to acronyms; `CCFMesh` loads
the region `.obj` meshes and builds k3d line objects for 3D plotting. Both read
the mounted `.brainglobe` data asset, so nothing is downloaded at run time.

This module sits beside the notebooks and is imported directly
(`from ccf_utils import CCF, CCFMesh`) -- no package install needed.
"""
import os

import k3d
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas


class CCF:
    """
    Common Coordinate Framework (CCF) region lookup.

    Exposes the voxel `resolution` and a region-id -> acronym map
    (`acronymMap`), sourced from the brainglobe `allen_mouse_25um` atlas -- the
    same atlas the region meshes come from (see `CCFMesh`). The atlas is read
    from the mounted `.brainglobe` data asset (`brainglobe_dir`), so nothing is
    downloaded at run time. See the README for how to obtain the atlas if that
    asset is unavailable.
    """

    def __init__(self, resolution=25, brainglobe_dir="/data/.brainglobe"):
        self.resolution = resolution
        atlas = BrainGlobeAtlas(
            f"allen_mouse_{resolution}um",
            brainglobe_dir=brainglobe_dir,
            check_latest=False,  # use the mounted atlas; do not hit the network
        )
        # lookup_df columns: acronym, id, name -> map region id to acronym
        self.acronymMap = dict(zip(atlas.lookup_df["id"], atlas.lookup_df["acronym"]))

    def __str__(self):
        prop_list = [prop for prop in dir(self) if not prop.startswith("__")]
        return "ccf has properties:\n" + "\n".join(prop_list)


class CCFMesh:
    @staticmethod
    def load_obj(filename):
        """
        Load the vertices, vertex normals, and indices from a .obj file.

        Parameters:
        filename (str): Path to the .obj file

        Returns:
        tuple: A tuple containing three elements:
            - vertices (list of tuples): List of vertices, each vertex is a tuple (x, y, z)
            - normals (list of tuples): List of vertex normals, each normal is a tuple (nx, ny, nz)
            - indices (list of tuples): List of indices, each index is a tuple of vertex indices defining a face
        """
        vertices = []
        normals = []
        indices = []

        with open(filename, "r") as file:
            for line in file:
                if line.startswith("v "):  # Vertex definition
                    parts = line.split()
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("vn "):  # Vertex normal definition
                    parts = line.split()
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("f "):  # Face definition
                    parts = line.split()
                    # Extracting only the vertex indices (ignoring texture and normal indices)
                    face_indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    indices.append(tuple(face_indices))

        return vertices, normals, indices

    @staticmethod
    def get_mesh_from_id(allen_id):
        obj_dir = "/data/.brainglobe/allen_mouse_25um_v1.2/meshes"
        obj_path = os.path.join(obj_dir, f"{allen_id}.obj")
        return CCFMesh.load_obj(obj_path)

    @staticmethod
    def rgb_to_hex(r, g, b):
        # Convert to a hexadecimal string
        hex_color = f"{r:02x}{g:02x}{b:02x}"
        # Convert the hexadecimal string to an integer in base-16
        color_int = int(hex_color, 16)
        return color_int

    @staticmethod
    def plot_graphs(graphs, plot, color=0):
        for i, g in enumerate(graphs):
            g_lines = CCFMesh.graph_to_lines(g, color)
            plot += g_lines

    @staticmethod
    def graph_to_lines(g, color):
        # Extract vertex positions
        g_verts = np.array([g.nodes[n]["pos"] for n in sorted(g.nodes())], dtype=np.float32)
        # Pairs of indices into the vertex array are edges
        # Node keys start at 1, so offset by -1 to get indices
        g_inds = np.array([[u - 1, v - 1] for u, v in g.edges()], dtype=np.float32)
        g_lines = k3d.factory.lines(g_verts, g_inds, indices_type="segment", color=color, width=1, shader="simple")
        return g_lines
