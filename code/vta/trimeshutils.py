import numpy as np
from pathlib import Path
from vta.utils import Brain, CCF, CCFMesh
from trimesh import Trimesh
from trimesh import load as load_trimesh
from multiprocessing import Pool, cpu_count


def process_chunk(chunk, mesh):
    return mesh.contains(points=chunk)


def trimesh_to_array(obj_file="", save_array_to=None):
    """
    Generate an array compatible with ccf parcellation to locate the volume specified by the mesh structure.

    Example: trimesh_to_array(obj_file="/root/capsule/results/LC_ccf_v1_250102.obj",
                              save_array_to="/root/capsule/results/LC_ccf_v1_250102_mask.npy")
    Args:
        obj_file: path to file created with `trimesh`.
        save_array_to: string for path to save the array.
    Returns:
        roi_mask: 3d numpy array.
    """

    # get ccf metadata
    ccf = CCF(reference_space_key="annotation/ccf_2017", output_dir="/results/")

    # native CCF V3 mask - this is just used to get the shape.
    roi_mask = ccf.get_roi_mask(roi_list=["LC"], mask_dilate_iterations=0)

    # load custom mesh
    mesh = load_trimesh(obj_file)

    mesh_verts = np.array(mesh.vertices)
    min_vals = np.min(mesh_verts, axis=0).astype(int)
    max_vals = np.max(mesh_verts, axis=0).astype(int)
    print("Bounds of the mesh object")
    print(min_vals)
    print(max_vals)

    # Indices of voxels within the bounts
    coords = np.stack(
        np.meshgrid(
            np.arange(min_vals[0], max_vals[0]),
            np.arange(min_vals[1], max_vals[1]),
            np.arange(min_vals[2], max_vals[2]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    print(f"Number of points to check: {coords.shape[0]}")
    print(f"Estimated time: {45*coords.shape[0]/50000/60:0.2f} minutes")

    # all we actually want to do is mesh.contains(points=coords)
    # we cropped the volume we check for, and then use multiprocessing to speed up the calculations.

    # Set chunk size
    chunk_size = 1000
    n_points = coords.shape[0]

    # Split coordinates into chunks
    chunks = [coords[i : i + chunk_size] for i in range(0, n_points, chunk_size)]

    # Use multiprocessing to process chunks in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_chunk, [(chunk, mesh) for chunk in chunks])

    # gather results into a single array
    inside = np.concatenate(results)

    new_roi_mask = np.full(roi_mask.shape, False)
    x, y, z = coords[inside].T  # unpack coordinates for indexing
    new_roi_mask[x, y, z] = True

    # saves the roi_mask as a 3d array.
    if save_array_to is not None:
        np.save(save_array_to, new_roi_mask)
        print(f"Saved file to: {save_array_to}")
    return new_roi_mask
