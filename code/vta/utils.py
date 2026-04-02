from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from dask import array as da
from scipy import ndimage
import os
import numpy as np
import k3d


class Brain:
    """
    A class to load and handle data from a whole brain volume.
    """

    baseResolution = [1.8, 1.8, 2]  # microns
    zarrMultiple = {j: 2**j for j in range(5)}  # compression at each zarr level
    injectionSites = {}  # injection site information, populated by getInjectionSites call

    def __init__(self, sample, level=3, verbose=True):
        self.verbose = verbose
        self.chPaths = None
        self.segPaths = None
        self.ccfCellsPaths = None

        # fetch data
        self.sample = str(sample)
        self.getPath()
        self.setLevel(level)
        self.setColorMaps()

    def __str__(self):
        # Print out the channels, cell segmentations, and CCF aligned quantifications found
        out_brain = f"Brain: {self.sample}\n"
        out_channels = f"Channels found: {list(self.chPaths.keys())}\n"
        out_cell_segmentations = f"Cell segmentations found: {list(self.segPaths.keys())}\n"
        out_ccf_aligned_quantifications = f"CCF aligned quantifications found: {list(self.ccfCellsPaths.keys())}\n"
        return out_brain + out_channels + out_cell_segmentations + out_ccf_aligned_quantifications

    # Methods
    def getPath(self, root="/data"):
        """Method to get path to whole brain volume data"""
        rootDir = Path(root)
        rootDir = [file for file in rootDir.iterdir() if self.sample in str(file)]

        # Check that the appropriate number of folders were found.
        if len(rootDir) > 1:
            raise ValueError("Found multiple directories matching requested sample ID.")
        elif len(rootDir) == 0:
            raise ValueError("Could not find a data directory matching input sample ID.")
        self.rootDir = rootDir[0]

        # Handle iteration of several formatting conventions
        sampleDir = rootDir[0].joinpath("processed", "stitching", "OMEZarr")
        if not sampleDir.exists():
            sampleDir = rootDir[0].joinpath("processed", "OMEZarr")
            if not sampleDir.exists():
                sampleDir = rootDir[0].joinpath("image_tile_fusing", "OMEZarr")

        if self.verbose:
            print(f"Loading data from {sampleDir}")

        # Grab channel, named by excitation
        chPaths = {exCh.name.split("_")[1]: exCh for exCh in sampleDir.glob("Ex*.zarr")}
        self.channels = list(chPaths.keys())
        self.chPaths = chPaths
        if self.verbose:
            print(f"Found the following channels: {self.channels}")

        # Grab cell segmentations
        segDir = rootDir[0].joinpath("image_cell_segmentation")
        segPaths = {exCh.name.split("_")[1]: exCh.joinpath("detected_cells.xml") for exCh in segDir.glob("Ex*")}
        self.segPaths = segPaths
        if self.verbose:
            print(f"Found cell segmentations in the following channels: {list(segPaths.keys())}")

        # Grab CCF quantifications
        quantDir = rootDir[0].joinpath("image_cell_quantification")
        quantPaths = {
            exCh.name.split("_")[1]: exCh.joinpath("cell_count_by_region.csv") for exCh in quantDir.glob("Ex*")
        }
        ccfCellsPaths = {
            exCh.name.split("_")[1]: exCh.joinpath("transformed_cells.xml") for exCh in quantDir.glob("Ex*")
        }
        self.quantPaths = quantPaths
        self.ccfCellsPaths = ccfCellsPaths
        if self.verbose:
            print(f"Found CCF aligned quantifications in the following channels: {list(quantPaths.keys())}")

    def setLevel(self, level):
        # Method to update level and grab hierarchical volume for corresponding resolution level
        self.level = level
        if self.verbose:
            print(f"Grabbing volumes for level: {level}")
        self.getVol()

    def getVol(self):
        # Method to mount volumetric imaging data
        self.vols = {
            channel: da.from_zarr(str(chPath), self.level).squeeze() for channel, chPath in self.chPaths.items()
        }

    def orientVol(self, ch, plane="coronal", returnLabels=False):
        """Method to orient requested channel volume to a particular plane. Return labels for internal methods, e.g. plotSlice"""
        if (plane.lower() == "horizontal") | (plane.lower() == "transverse"):
            printTxt = "Plotting horizontal axis, "
            axis = 0
            xLabel = "M/L"
            yLabel = "A/P"
        elif plane.lower() == "sagittal":
            printTxt = "Plotting sagittal axis, "
            axis = 2
            xLabel = "A/P"
            yLabel = "D/V"
        else:
            plane = "coronal"
            printTxt = "Plotting coronal axis, "
            axis = 1
            xLabel = "M/L"
            yLabel = "D/V"
        chVol = da.moveaxis(self.vols[ch], axis, 0)

        if returnLabels:
            return chVol, xLabel, yLabel, printTxt
        else:
            return chVol

    def setColorMaps(self, base="black", channelColors={}):
        # Method to establish each channel's color map for future plotting. Modifies default colors via channelColors channel:color dictionary pairs
        colorSets = {
            "445": "turquoise",
            "488": "lightgreen",
            "561": "tomato",
            "639": "white",
        }  # default colors
        colormaps = {}

        # Modify color sets if channel colors are provided
        for ch, color in channelColors.items():
            if ch not in self.channels:
                raise ValueError(f"Trying to set color for channel {ch}, but channel was not found in dataset.")
            else:
                colorSets[ch] = color

        # Generate color maps for channels present in data
        for ch in self.channels:
            if ch not in colorSets.keys():
                print(f"No default color exists for the {ch} channel, setting to white.")
                colormaps[ch] = sns.blend_palette([base, "white"], as_cmap=True)
            else:
                colormaps[ch] = sns.blend_palette([base, colorSets[ch]], as_cmap=True)
        self.colormaps = colormaps

    def getInjectionSite(self, ch, level=4, plane="sagittal", span=60, showPlot=True):
        # Method to localize viral injection sites.
        self.setLevel(level, showPlot)
        # For a given channel, find the center of mass in a span around the brightest point in the volume.
        chVol = self.orientVol(ch, plane=plane)  # Think about best orientation to save coordinates in
        posMax = np.argmax(chVol).compute()  # Find brightest pixel in entire volume, then convert to index
        indxMax = np.unravel_index(posMax, chVol.shape)
        # Further process on volume centered at brightest point, size governed by span
        xSlice, ySlice, zSlice = (
            slice(indxMax[0] - span, indxMax[0] + span),
            slice(indxMax[1] - span, indxMax[1] + span),
            slice(indxMax[2] - span, indxMax[2] + span),
        )
        centerVol = chVol[xSlice, ySlice, zSlice]
        # Clip volume to signal for CoM calculation
        clipVals = np.quantile(centerVol, [0.95, 0.995])
        centerVol = centerVol - clipVals[0]  # Set everything below 95% to 0, clip to 95th percentile
        centerVol = centerVol.clip(0, clipVals[1] - clipVals[0])
        com = np.round(ndimage.center_of_mass(np.array(centerVol)))
        # Plot if requested
        if showPlot:
            plt.imshow(centerVol[com[0], :, :], cmap=self.colormaps[ch], vmax=1200)
            plt.plot(com[2], com[1], "or")

        coord = com - span + indxMax
        self.injectionSites[ch] = {
            "plane": plane,
            "level": level,
            "span": span,
            "coordinates": coord,
        }
        # Edit later to include fitting function
        return centerVol

    def plotSlice(
        self,
        ch=[],
        plane="coronal",
        section=[],
        extent=[],
        level=3,
        vmin=0,
        vmax=600,
        alpha=1,
        ticks=True,
        printOutput=True,
    ):
        # Method to plot a particular slice

        # If no channel is provided, plot shortest wavelength
        if not ch:
            ch = min(self.channels)

        # Specify resolution level, and then retrieve properly oriented volume
        self.setLevel(level, printOutput)
        [chVol, xLabel, yLabel, printTxt] = self.orientVol(ch, plane=plane, returnLabels=True)

        # Get data indices to be plotted
        if not section:
            sectionIndex = int(chVol.shape[0] / 2)
            section = sectionIndex * self.zarrMultiple[level]
        else:  # otherwise convert microns to indices
            sectionIndex = int(section / self.zarrMultiple[level])
        if (not extent) | len(extent) != 4:
            extentIndices = np.array([0, chVol.shape[2], chVol.shape[1], 0])
            extent = extentIndices * self.zarrMultiple[level]
        else:  # interpret extent requests as microns, convert to indices
            extentIndices = np.round(np.array(extent) / self.zarrMultiple[level])
        if printOutput:
            print(printTxt + "secion: " + str(section) + " (level " + str(level) + " index: " + str(sectionIndex) + ")")

        # Plot data
        plt.imshow(
            chVol[
                sectionIndex,
                extentIndices[3] : extentIndices[2],
                extentIndices[0] : extentIndices[1],
            ],
            cmap=self.colormaps[ch],
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            alpha=alpha,
            interpolation="none",
        )
        if ticks:
            plt.title(ch)
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
        else:
            plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    def plotPoint(self, cst, ch: list = [], span=20, vmin=0, vmax=600):
        # Method to plot a given point in 3 planes, specified by variable cst (coronal, sagittal, transverse).

        # If no channel is provided, plot shortest wavelength
        if not ch:
            ch = min(self.channels)

        if span > 300:
            level = 1
        else:
            level = 0

        # Set up subplots
        nChannels = len(ch)
        planeDict = {0: "Coronal", 1: "Sagittal", 2: "Transverse"}
        extentDict = {
            0: [cst[1] - span, cst[1] + span, cst[2] + span, cst[2] - span],  # M/L, D/V
            1: [cst[0] - span, cst[0] + span, cst[2] + span, cst[2] - span],  # A/P, D/V
            2: [cst[1] - span, cst[1] + span, cst[0] + span, cst[0] - span],  # M/L, A/P
        }
        for chIndx, channel in enumerate(ch):
            for planeIndx in range(3):
                plt.subplot(nChannels, 3, 1 + planeIndx + chIndx * 3)
                self.plotSlice(
                    ch=channel,
                    plane=planeDict[planeIndx],
                    section=cst[planeIndx],
                    extent=extentDict[planeIndx],
                    level=0,
                    vmin=vmin,
                    vmax=vmax,
                    printOutput=False,
                    ticks=False,
                )
                if chIndx == 0:
                    plt.title(planeDict[planeIndx])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

    def plotBlend(
        self,
        ch: list = [],
        plane="coronal",
        section=[],
        extent=[],
        level=3,
        alphaDict=[],
        vDict=[],
        ticks=True,
    ):
        # Method to plot blended channels

        # If no channels are provided, plot all. Default to longest wavelength first
        if not ch:
            ch = self.channels
            if ch[0] < ch[-1]:
                ch = ch[::-1]

        # If no alpha dict values are provided, use default
        if not alphaDict:
            defaultAlpha = 1 / len(ch)
            alphaDict = {channel: defaultAlpha for channel in ch}

        # If no vmin / vmax dict values are provided, use default
        if not vDict:
            vDict = {channel: [0, 600] for channel in ch}

        # Plot blended channels
        for channel in ch:
            self.plotSlice(
                ch=channel,
                plane=plane,
                section=section,
                extent=extent,
                level=level,
                alpha=alphaDict[channel],
                vmin=vDict[channel][0],
                vmax=vDict[channel][1],
                printOutput=False,
                ticks=ticks,
            )
        plt.title("")

    def getNGLink(self):
        # Method to print neuroglancer link of associated imaging data
        linkPath = self.rootDir.joinpath("neuroglancer_config.json")
        # linkPath =self.rootDir.joinpath("image_cell_segmentation/Ex_561_Em_593/visualization/neuroglancer_config.json")
        ngJSON = pd.read_json(linkPath, orient="index")
        print(ngJSON[0]["ng_link"])

    def getCellsCCF(self, ch: list):
        # Method to retrieve and format CCF transformed coordinates of segemented cells
        locationDict = {}
        for channel in ch:
            locCellsDF = pd.read_xml(
                self.ccfCellsPaths[channel],
                xpath="//CellCounter_Marker_File//Marker_Data//Marker_Type//Marker",
            )
            locationDict[channel] = locCellsDF

        return locationDict


class CCF:
    """
    A class to handle Common Coordinate Framework (CCF) operations.
    """

    def __init__(
        self,
        resolution=25,
        reference_space_key="annotation/ccf_2017",
        output_dir="/results/",
    ):
        self.resolution = resolution
        self.reference_space_key = reference_space_key
        self.rspc = ReferenceSpaceCache(
            self.resolution,
            self.reference_space_key,
            manifest=Path(output_dir) / "manifest.json",
        )
        self.annot, self.annot_info = self.rspc.get_annotation_volume()
        self.structure_tree = self.rspc.get_structure_tree()
        self.rsp = self.rspc.get_reference_space()

        # Set up CCF structure lookup dicts
        idMap = self.structure_tree.get_id_acronym_map()
        self.acronymMap = {IDs: acronym for acronym, IDs in idMap.items()}  # invert idMap
        return

    def __str__(self):
        prop_list = [prop for prop in dir(self) if not prop.startswith("__")]
        out = "\n".join(prop_list)
        return f"ccf has properties:\n" + out

    def get_roi_mask(self, roi_list=["LC"], mask_dilate_iterations=10):
        roi_map = self.structure_tree.get_id_acronym_map()
        id_list = [roi_map[roi] for roi in roi_list]
        roi_mask = self.rsp.make_structure_mask(id_list).astype("bool")  # in shape AP, DV, ML
        roi_mask = ndimage.binary_dilation(
            roi_mask, iterations=mask_dilate_iterations
        )  # dilate LC annotation by 10 iterations * 25 um = 250 um
        return roi_mask


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
        obj_dir = "/data/ccf_2017_obj"
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
