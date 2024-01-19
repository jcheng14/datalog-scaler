# ----------------------------------------------------------------------------
# - Pointcloud Handler Class using Open3D                                    -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
from __future__ import annotations
import copy
import numpy as np
from typing import Tuple
from varname import nameof
import open3d as o3d
import open3d.visualization.gui as gui  # noqa
import matplotlib.pyplot as plt
import utility as ut
from utility import FloatArray3, FloatArray4, FloatArrayN, IntArrayN  # noqa
from utility import FloatArray3x3, FloatArrayNx3, BoolArray3, BoolArrayN  # noqa
from utility import PointCloud, TriangleMesh, OrientedBoundingBox  # noqa


class PointcloudHandler:
    """
    PointcloudHandler class definition

    """

    def __init__(
        self,
        pcd_raw: PointCloud,
        pcd_idx: list[int],
        vec2sky: FloatArray3 = ut.zaxis,
        vec2snr: FloatArray3 = -ut.yaxis,
        disp_info: bool = True,
        disp_progress: bool = True,
    ) -> None:
        """
        Construction function for the PointcloudHandler class

        Args:
            pcd_raw(type: o3d.PointCloud) -- the raw point cloud
            pcd_idx(type: list[int]) -- pcd's index list, initial all -1,
                but will be setting sensor-, ground- and reference- plane pcd
                indices during the data processing
            vec2sky(type: np.ndarray) -- pcd's upward vec (default zaxis)
            vec2snr(type: np.ndarray) -- pcd to sensor vec (default -yaxis)
            disp_info(type: bool) -- display debug info (dafault False)
            disp_progress(type: bool) -- display progress info (dafault False)
        """
        # private parameters for dbscan and ransac clustering
        # these can be packaged into a yaml config file later.
        self.__dbscan_epsilon_lo: float = 0.0005
        self.__dbscan_epsilon_hi: float = 0.5
        self.__dbscan_epsilon_default: float = 0.005
        self.__dbscan_epsilon: float = 0.005
        self.__distance_threshold_lo: float = 0.002
        self.__distance_threshold_hi: float = 0.5
        self.__distance_threshold_default: float = 0.01
        self.__distance_threshold: float = 0.01
        self.__min_cluster_pts: int = 200

        # private parameters for labeling indices for the pcds in the
        # different detected/synthethized entities -- general pcd (0),
        # ground_obox (-1), reference_obox (-2), sensor_obox (-3),
        # then individual planar patch labels starting from 0 in
        # planar_patch detection, and plane colors, uvw tri-axes scale
        self.__gnr_pcd_idx_label: int = -1
        self.__grd_pcd_idx_label: int = -2
        self.__snr_pcd_idx_label: int = -3
        self.__ref_pcd_idx_label: int = -4
        self.__gnr_color : FloatArray3 = ut.CP.GRAY
        self.__grd_plane_color: FloatArray3 = ut.CP.GRAY_DARK
        self.__snr_plane_color: FloatArray3 = ut.CP.CYAN_DARK
        self.__snr_arr_color: FloatArray3 = ut.CP.ORANGE_DARK
        self.__ref_plane_color: FloatArray3 = ut.CP.RED_DARK
        self.__uvw_scale: float = 0.5
        self.__pcd_nv2snr: PointCloud

        # Sanity check for the input arguments
        ut.sck.is_valid_pcd(pcd_raw, "pcd_raw")
        ut.sck.is_nonzero_vector3(vec2sky, nameof(vec2sky))
        ut.sck.is_nonzero_vector3(vec2snr, nameof(vec2snr))

        # constructor parameters
        self.pcd_raw: PointCloud = pcd_raw
        self.pcd_idx: list[int]
        if pcd_idx is None or len(pcd_idx) == 0:
            self.pcd_idx = [self.__gnr_pcd_idx_label] * len(pcd_raw)
        else:
            assert len(pcd_idx) == len(pcd_raw.points), (
                f"[ERROR: {ut.currentFuncName()}]: "
                f"pcd_idx and pcd_raw array length mismatch!"
            )
            self.pcd_idx = pcd_idx
        self.vec2sky: FloatArray3 = vec2sky
        self.vec2snr: FloatArray3 = vec2snr

        # for Photogrammrtry PCD, a scaling factor is needed to scale the
        # pcd to real physical dimensions, this needs to be automated later
        # by picking up and measuring the fiducial(s) with known physical
        # dimensions -- can write a estimate_phtgm2real_scale() member func
        # for assgining the scaling factor to a member variable 
        # self.phtgm2real_scale: float = phtgm2real_scale

        # control info (for debug) and progress (for process) message display
        self.disp_info: bool = disp_info
        self.disp_progress: bool = disp_progress

        # make memory copy of pcd_raw, pcd is the working variable
        self.pcd: PointCloud = copy.deepcopy(pcd_raw)

        # log-end diamters_long|short, and distances to the reference plane,
        # nvecs, oboxes, segments, meshes. These are the end-points for
        # detection and engineering calculations in SI units.
        # See self.logend_scaling_report for scaling report in English metrics
        self.logend_diameters_long: list[float]
        self.logend_diameters_short: list[float]
        self.logend_dist2ref: list[float]
        self.logend_nvecs: list[FloatArray3]
        self.logend_centers: list[FloatArray3]
        self.logend_oboxes: list[OrientedBoundingBox]
        self.logend_segments: list[PointCloud]
        self.logend_meshes: list[TriangleMesh]

        # scaling report items in English metrics
        self.logend_scaling_report: list[dict]

        # ground plane parameters
        self.grd_plane_model: FloatArray4
        self.grd_plane_segment: PointCloud
        self.grd_plane_obox: OrientedBoundingBox
        self.grd_plane_mesh: TriangleMesh

        # sensor plane parameters, and sensor arr meshes
        self.snr_plane_model: FloatArray4
        self.snr_plane_segment: PointCloud
        self.snr_plane_obox: OrientedBoundingBox
        self.snr_plane_mesh: TriangleMesh
        self.snr_arr_mesh: TriangleMesh

        # fiducial planes parameters -- could be multiple fiducials
        self.fid_plane_models: list[FloatArray4]
        self.fid_segments: list[PointCloud]
        self.fid_oboxes: list[OrientedBoundingBox]
        self.fid_meshes: list[TriangleMesh]

        # reference plane is assembled from fiducials
        self.ref_plane_model: FloatArray4
        self.ref_plane_segment: PointCloud
        self.ref_plane_obox: OrientedBoundingBox
        self.ref_plane_mesh: TriangleMesh

        # eculidean_dbscan results
        self.labels_eculidean_dbscan: list[int]

        # ransac_dbscan clustering results
        self.ransac_dbscan_models: list[FloatArray4]
        self.ransac_dbscan_segments: list[PointCloud]
        self.ransac_dbscan_oboxes: list[OrientedBoundingBox]
        self.ransac_dbscan_rest: PointCloud

        # ransac_plane clustering results
        self.ransac_plane_models: list[FloatArray4]
        self.ransac_plane_segments: list[PointCloud]
        self.ransac_plane_oboxes: list[OrientedBoundingBox]
        self.ransac_plane_rest: PointCloud

        # planar_patch clustering results
        self.planar_patch_segments: list[PointCloud]
        self.planar_patch_oboxes: list[OrientedBoundingBox]
        self.planar_patch_selected_idx: list[bool]

    def set_pcd_idx_label(self, selected_idx: list[int], label: int) -> None:
        """
        Set the pcd_idx labels according to the given selected_idx list and
            the given label value for a single cluster within the point cloud
        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")

        if selected_idx is None or len(selected_idx) == 0:
            self.pcd_idx = [self.__gnr_pcd_idx_label] * len(self.pcd.points)
        else:
            assert len(selected_idx) <= len(self.pcd.points), (
                f"[ERROR: {ut.currentFuncName()}]: "
                f"the selected_idx list is longer than the pcd array length!"
            )
            pcd_idx_np = np.array(self.pcd_idx)
            pcd_idx_np[selected_idx] = label
            self.pcd_idx = pcd_idx_np.tolist()

    def set_pcd_color(
        self, selected_idx: list[int], paint_colors: FloatArrayNx3 | FloatArray3
    ) -> None:
        """
        Set the pcd colors according to the given selected_idx list and
            the given color value or the color value list for the point cloud
        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")

        assert (len(selected_idx)) > 0 & (
            len(selected_idx) <= len(self.pcd.points)
        ), (
            f"[ERROR: {ut.currentFuncName()}]: "
            f"the selected_idx list is longer than the pcd array length!"
        )
        assert (len(paint_colors)) > 0 & (
            len(paint_colors) <= len(self.pcd.points)
        ), (
            f"[ERROR: {ut.currentFuncName()}]: "
            f"the paint_color array is longer than the pcd array length!"
        )

        np_colors = np.array(self.pcd.colors)
        if not self.pcd.has_colors():
            self.pcd.colors = o3d.utility.Vector3dVector(
                np.tile(self.__gnr_color, (len(self.pcd.points), 1))

            )
        np_colors = np.array(self.pcd.colors)
        if len(paint_colors) == 1:
            np_colors[selected_idx, :] = np.tile(
                paint_colors, (len(selected_idx), 1)
            )
        else:
            np_colors[selected_idx, :] = paint_colors

        self.pcd.colors = o3d.utility.Vector3dVector(np_colors)

    def get_dbscan_epsilon(self) -> float:
        return self.__dbscan_epsilon

    def set_dbscan_epsilon(self, dbscan_epsilon: float) -> None:
        ut.sck.is_float_value_bounded(
            dbscan_epsilon,
            self.__dbscan_epsilon_lo,
            self.__dbscan_epsilon_hi,
            "dbscan_epsilon",
        )
        self.__dbscan_epsilon = dbscan_epsilon

    def get_dbscan_epsilon_default(self) -> float:
        return self.__dbscan_epsilon_default

    def set_dbscan_epsilon_default(self, dbscan_epsilon_default: float):
        ut.sck.is_float_value_bounded(
            dbscan_epsilon_default,
            self.__dbscan_epsilon_lo,
            self.__dbscan_epsilon_hi,
            "dbscan_epsilon_default",
        )
        self.__dbscan_epsilon_default = dbscan_epsilon_default

    def get_dbscan_epsilon_bound(self) -> Tuple[float, float]:
        return (self.__dbscan_epsilon_lo, self.__dbscan_epsilon_hi)

    def set_dbscan_epsilon_bound(
        self, dbscan_epsilon_lo: float, dbscan_epsilon_hi: float
    ):
        ut.sck.is_float_hi_gt_lo(
            hi=dbscan_epsilon_hi,
            lo=dbscan_epsilon_lo,
            hi_name="dbscan_epsilon_hi",
            lo_name="dbscan_epsilon_lo",
        )
        self.__dbscan_epsilon_lo = dbscan_epsilon_lo
        self.__dbscan_epsilon_hi = dbscan_epsilon_hi

    def get_distance_threshold(self) -> float:
        return self.__distance_threshold

    def set_distance_threshold(self, distance_threshold: float):
        ut.sck.is_float_value_bounded(
            distance_threshold,
            self.__distance_threshold_lo,
            self.__distance_threshold_hi,
            "__distance_threshold",
        )
        self.__distance_threshold = distance_threshold

    def get_distance_threshold_default(self) -> float:
        return self.__distance_threshold_default

    def set_distance_threshold_default(self, distance_threshold_default: float):
        ut.sck.is_float_value_bounded(
            distance_threshold_default,
            self.__distance_threshold_lo,
            self.__distance_threshold_hi,
            "distance_threshold_default",
        )
        self.__distance_threshold_default = distance_threshold_default

    def get_distance_threshold_bound(self) -> Tuple[float, float]:
        return (self.__distance_threshold_lo, self.__distance_threshold_hi)

    def set_distance_threshold_bound(
        self, distance_threshold_lo: float, distance_threshold_hi: float
    ) -> None:
        ut.sck.is_float_hi_gt_lo(
            hi=distance_threshold_hi,
            lo=distance_threshold_lo,
            hi_name="distance_threshold_hi",
            lo_name="distance_threshold_lo",
        )
        self.__distance_threshold_lo = distance_threshold_lo
        self.__distance_threshold_hi = distance_threshold_hi

    def remove_statistical_outlier(
        self, nb_neighbors: int = 30, std_ratio: float = 0.5
    ) -> Tuple[PointCloud, list[int]]:
        """
        Remove outliers using statistical noise rejection

        Args:
            nb_neighbors(type: int)=30 (default)
                -- number of neighboring pts, positive int
            std_ratio(type: float)=0.5 (default)
                -- the stand deviation ratio threshold for the rejection,
                -- positive float

        Returns:
            cl(type: o3d.geometry.PointCloud) -- the resulting pcd
            cl_ind(type: list[int]) -- the index list of the resulting pcd

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(nb_neighbors, nameof(nb_neighbors))
        ut.sck.is_positive_float(std_ratio, nameof(std_ratio))

        cl, cl_ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        cl.estimate_normals()
        return (cl, cl_ind)

    def remove_radius_outlier(
        self, nb_points: int = 30, radius: float = 0.01
    ) -> Tuple[PointCloud, list[int]]:
        """
        Remove outliers using radius neighbor search & rejection method

        Args:
            nb_points(type: int)=30 (default)
                -- number of neighboring pts, positive int
            radius(type: float)=0.01 (default)
                -- the radius distance for search & rejection, positive float

        Returns:
            cl(type: o3d.geometry.PointCloud) -- the resulting pcd
            cl_ind(type: list[int]) -- the index list of the resulting pcd

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(nb_points, nameof(nb_points))
        ut.sck.is_positive_float(radius, nameof(radius))

        cl, cl_ind = self.pcd.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        cl.estimate_normals()
        return (cl, cl_ind)

    def down_sample_voxel_size(
        self, ds_voxel_size: float = 0.005
    ) -> PointCloud:
        """
        Downsample the point cloud using a give voxel size

        Args:
            ds_voxel_size(type: float)=0.005 (default)
                -- the given voxel size, positive float

        Returns:
            cl(type: o3d.geometry.PointCloud) -- the resulting pcd

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_float(ds_voxel_size, nameof(ds_voxel_size))
        cl = self.pcd.voxel_down_sample(voxel_size=ds_voxel_size)
        cl.estimate_normals()
        return cl

    def down_sample_uniform(self, every_k_point: int = 10) -> PointCloud:
        """
        Downsample the point cloud using a uniform k_points decimation

        Args:
            every_k_point(type: int)=10 (default)
                -- the number of sparsing/decimatiing points, positive int

        Returns:
            cl(type: o3d.geometry.PointCloud) -- the resulting pcd

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(every_k_point, nameof(every_k_point))
        cl = self.pcd.uniform_down_sample(every_k_points=every_k_point)
        cl.estimate_normals()
        return cl

    def euclidean_dbscan(
        self, dbscan_epsilon: float, npt_min_cluster: int = 10
    ) -> None:
        """
        Conduct DBSCAN Euclidean clustering to segment the point cloud

        Args:
            dbscan_epsilon(type: float)
            npt_min_cluster(type: int)=10 (default)

        Returns:
            None

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(npt_min_cluster, nameof(npt_min_cluster))

        if dbscan_epsilon is None:
            self.set_dbscan_epsilon(self.__dbscan_epsilon_default)
        else:
            self.set_dbscan_epsilon(dbscan_epsilon)
        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"set dbscan_epsilon = {self.__dbscan_epsilon:.2e}"
            )

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"start euclidean_dbscan with "
                f"eps={self.__dbscan_epsilon:.2e}, "
                f"min_points={npt_min_cluster}"
            )
        labels = self.pcd.cluster_dbscan(
            eps=self.__dbscan_epsilon,
            min_points=npt_min_cluster,
            print_progress=self.disp_info,
        )
        if self.disp_progress:
            if labels is not None and len(labels) > 0:
                self.labels_eculidean_dbscan = labels
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"point cloud has {labels.max() + 1} clusters"
                )
            else:
                self.labels_eculidean_dbscan = [0] * len(self.pcd.points)
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"point cloud has no detected clusters!"
                )

    def plane_ransac_dbscan(
        self,
        dbscan_epsilon: float = 0.001,
        npt_min_cluster: int = 10,
        npt_ransac: int = 3,
        distance_threshold: float = 0.01,
        num_ransac_iters: int = 300,
        max_num_planes: int = 50,
    ) -> None:
        """
        Conduct looped & refined plane RANSAC with DBSCAN Euclidean
        clustering to segment the point cloud

        Args:
            dbscan_epsilon(type: float)
            npt_min_cluster(type: int)=10 (default)
            npt_ransac(type: int)=3 (default)
            distance_threshold: float
            num_ransac_iters(type: int)=300 (default)
            max_num_planes(type: int)=50 (default)

        Returns:
            None

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(npt_min_cluster, nameof(npt_min_cluster))
        ut.sck.is_positive_int(npt_ransac, nameof(npt_ransac))
        ut.sck.is_positive_int(num_ransac_iters, nameof(num_ransac_iters))
        ut.sck.is_positive_int(max_num_planes, nameof(max_num_planes))

        # check dbscan_epsilon
        if dbscan_epsilon is None:
            self.set_dbscan_epsilon(self.__dbscan_epsilon_default)
        else:
            ut.sck.is_positive_float(dbscan_epsilon, nameof(dbscan_epsilon))
            self.set_dbscan_epsilon(dbscan_epsilon)
        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"set dbscan_epsilon = {self.__dbscan_epsilon:.2e}"
            )

        # check distance_threshold
        if distance_threshold is None:
            self.set_distance_threshold(self.__distance_threshold_default)
        else:
            ut.sck.is_positive_float(
                distance_threshold, nameof(distance_threshold)
            )
            self.set_distance_threshold(distance_threshold)
        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"set distance_threshold = {self.__distance_threshold:.2e}"
            )

        # start the ransac + dbscan loop
        plane_models = []
        segments = []
        rest = self.pcd
        nth_plane = 0
        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"start plane_ransac + dbscan loop with "
                f"eps={self.__dbscan_epsilon:.2e}, "
                f"min_points={npt_min_cluster}, "
                f"npt_ransac={npt_ransac}, "
                f"distance_threshold={self.__distance_threshold:.2e}, "
                f"num_ransac_iters={num_ransac_iters}, "
                f"max_num_planes={max_num_planes}"
            )
        while (rest.has_points()) & (nth_plane < max_num_planes):
            # plane_ransac
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.__distance_threshold,
                ransac_n=npt_ransac,
                num_iterations=num_ransac_iters,
            )
            nth_plane += 1
            if (
                plane_model is not None
                and np.all(np.abs(plane_model[0:2]) >= 1e-3)
                and inliers is not None
                and len(inliers) > 0
            ):
                # eculidean_dbscan on the choosen plane
                this_plane_seg = rest.select_by_index(inliers)
                if this_plane_seg is not None and this_plane_seg.has_points():
                    labels = np.array(
                        this_plane_seg.cluster_dbscan(
                            eps=self.__dbscan_epsilon,
                            min_points=npt_min_cluster,
                            print_progress=self.disp_info,
                        )
                    )
                    candidates = [
                        len(np.where(labels == j)[0]) for j in np.unique(labels)
                    ]
                    best_candidate = int(
                        np.array(
                            np.unique(labels)[
                                np.where(candidates == np.max(candidates))[0]
                            ]
                        )[0]
                    )
                    if self.disp_info:
                        print(
                            f"[INFO: {ut.currentFuncName()}]: "
                            f"the best cluster candidate is: {best_candidate}"
                        )

                    rest = rest.select_by_index(
                        inliers, invert=True
                    ) + this_plane_seg.select_by_index(
                        list(np.where(labels != best_candidate)[0])
                    )
                    seg = this_plane_seg.select_by_index(
                        list(np.where(labels == best_candidate)[0])
                    )
                    if seg.has_points():
                        segments.append(seg)
                        plane_models.append(plane_model)
                        if self.disp_info:
                            print(
                                f"[INFO: {ut.currentFuncName()}]: "
                                f"pass {nth_plane}/{max_num_planes} done, "
                                f"found plane eq: {plane_model[0]:+.2f}x "
                                f"{plane_model[1]:+.2f}y {plane_model[2]:+.2f}z"
                                f" {plane_model[3]:+.2f} = 0"
                            )
                    else:
                        if self.disp_info:
                            print(
                                f"[INFO: {ut.currentFuncName()}]: "
                                f"pass {nth_plane}/{max_num_planes} done, "
                                f"found plane, but no pcd cluster found"
                            )

        # get oboxes from the segments, and assign colors
        valid_idx = [seg.has_points() for seg in segments]
        if valid_idx is not None and len(valid_idx) > 0:
            segments = np.asarray(segments)[valid_idx].tolist()
            plane_models = np.asarray(plane_models)[valid_idx].tolist()
        oboxes = []
        for jj, seg in enumerate(segments):
            colors = plt.get_cmap("tab20")(jj)
            seg.paint_uniform_color(list(colors[:3]))
            obox = seg.get_oriented_bounding_box(robust=True)
            obox.color = list(colors[:3])
            oboxes.append(obox)

        # set the relevant results
        self.ransac_dbscan_models = plane_models
        self.ransac_dbscan_segments = segments
        self.ransac_dbscan_oboxes = oboxes
        self.ransac_dbscan_rest = rest

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"found {len(self.ransac_dbscan_oboxes)} planes"
            )

    def plane_ransac(
        self,
        npt_ransac: int = 3,
        distance_threshold: float = 0.01,
        num_ransac_iters: int = 300,
        max_num_planes: int = 50,
    ) -> None:
        """
        Conduct RANSAC loop to detect multiple planar shapes

        Args:
            npt_ransac(type: int)=3 (default)
            distance_threshold: float = 0.01 (default)
            num_ransac_iters(type: int)=300 (default)
            max_num_planes(type: int)=50 (default)

        Returns:
            None

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_int(npt_ransac, nameof(npt_ransac))
        ut.sck.is_positive_int(num_ransac_iters, nameof(num_ransac_iters))
        ut.sck.is_positive_int(max_num_planes, nameof(max_num_planes))

        if distance_threshold is None:
            self.set_distance_threshold(self.__distance_threshold_default)
        else:
            self.set_distance_threshold(distance_threshold)
        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"set distance_threshold = {self.__distance_threshold:.2e}"
            )

        plane_models = []
        segments = []
        oboxes = []
        rest = self.pcd
        plane_model = []
        nth_plane = 0
        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"start plane_ransac loop with npt_ransac={npt_ransac}, "
                f"distance_threshold={self.__distance_threshold:.2e}, "
                f"num_ransac_iters={num_ransac_iters}, "
                f"max_num_planes={max_num_planes}"
            )
        while (
            (rest.has_points())
            & (len(rest.points) >= self.__min_cluster_pts)
            & (nth_plane < max_num_planes)
        ):
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.__distance_threshold,
                ransac_n=npt_ransac,
                num_iterations=num_ransac_iters,
            )
            nth_plane += 1
            if (
                plane_model is not None
                and len(plane_model) == 4
                and np.all(np.abs(plane_model[0:2]) >= 1e-2)
                and inliers is not None
                and len(inliers) > self.__min_cluster_pts
            ):
                seg = rest.select_by_index(inliers)
                if (
                    seg is not None
                    and seg.has_points()
                    and len(seg.points) >= self.__min_cluster_pts
                ):
                    colors = plt.get_cmap("tab20")(nth_plane)
                    seg.paint_uniform_color(list(colors[:3]))
                    obox = seg.get_oriented_bounding_box(robust=True)
                    obox.color = list(colors[:3])
                    oboxes.append(obox)
                    segments.append(seg)
                    plane_models.append(np.array(plane_model))

                    rest = rest.select_by_index(inliers, invert=True)
                    if self.disp_info:
                        print(
                            f"[INFO: {ut.currentFuncName()}]: "
                            f"pass {nth_plane}/{max_num_planes} done, "
                            f"found plane eq: "
                            f"{plane_model[0]:+.2f}x {plane_model[1]:+.2f}y "
                            f"{plane_model[2]:+.2f}z {plane_model[3]:+.2f} = 0"
                        )
                else:
                    if self.disp_info:
                        print(
                            f"[INFO: {ut.currentFuncName()}]: "
                            f"pass {nth_plane}/{max_num_planes} done, found "
                            f"plane but no pcd cluster."
                        )
            else:
                if self.disp_info:
                    print(
                        f"[INFO: {ut.currentFuncName()}]: "
                        f"pass {nth_plane}/{max_num_planes} done, "
                        f"found no plane."
                    )

        # set the results to the relevant parameters
        self.ransac_plane_models = plane_models
        self.ransac_plane_segments = segments
        self.ransac_plane_oboxes = oboxes
        self.ransac_plane_rest = rest

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"found {len(self.ransac_plane_oboxes)} plane clusters"
            )

    def planar_patches(
        self,
        the_pcd: PointCloud,
        normal_variance_threshold_deg: float = 60.0,
        coplanarity_deg: float = 75.0,
        outlier_ratio: float = 0.75,
        min_plane_edge_length: float = 0.0,
        min_num_points: int = 0,
        search_knn: int = 30,
    ) -> None:
        """
        Conduct planar patch clustering to segment the point cloud

        Args:
            the_pcd(type: PointCloud)
                -- the pcd to be processed thru planar patch detector
            normal_variance_threshold_deg(type: float)=60.0 (default)
            coplanarity_deg(type: float)=75.0 (default)
            outlier_ratio(type: float)=0.75 (default)
            min_plane_edge_length(type: int)=0 (default)
            min_num_points(type: int)=0 (default)
            search_knn(type: int)=30 (default)

        Returns:
            N/A

        Reference -- http://www.open3d.org/docs/release/jupyter/geometry/pointcloud.html?highlight=detect_planar_patches#Planar-patch-detection     # noqa

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_valid_pcd(the_pcd, "the_pcd")

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"start planar_patches_detection with "
                f"normal_variance_threshold_deg="
                f"{normal_variance_threshold_deg:.2f}, "
                f"coplanarity_deg={coplanarity_deg:.2f}, "
                f"outlier_ratio={outlier_ratio:.2f}, "
                f"min_plane_edge_length={min_plane_edge_length:.2e}, "
                f"min_num_points={min_num_points}, "
                f"search_knn={search_knn}"
            )

        oboxes = the_pcd.detect_planar_patches(
            normal_variance_threshold_deg=normal_variance_threshold_deg,
            coplanarity_deg=coplanarity_deg,
            outlier_ratio=outlier_ratio,
            min_plane_edge_length=min_plane_edge_length,
            min_num_points=min_num_points,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=search_knn),
        )

        # get segments from oboxes
        segments = []
        for jj, this_obox in enumerate(oboxes):
            # flip the obox along its uu if its ww vector aligns to -vec2snr
            obox, _, _ = ut.flip_obox(this_obox, self.vec2snr)
            colors = plt.get_cmap("tab20")(jj)
            obox.color = list(colors[:3])
            idx = obox.get_point_indices_within_bounding_box(self.pcd.points)
            seg = self.pcd.select_by_index(idx)
            if seg.has_points():
                seg.paint_uniform_color(list(colors[:3]))
                segments.append(seg)

        # take out oboxes with no pcd, if any
        valid_idx = [seg.has_points() for seg in segments]
        if valid_idx is not None and len(valid_idx) > 0:
            oboxes = np.asarray(oboxes)[valid_idx].tolist()

        # set the relevant results
        self.planar_patch_segments = segments
        self.planar_patch_oboxes = oboxes

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"found {len(self.planar_patch_oboxes)} planar patches"
            )

    def get_grd_plane(
        self,
        pose_ang_lo: float = np.pi / 6,
        pose_ang_hi: float = np.pi / 6 * 5,
    ) -> None:
        """
        Detect the ground plane in the point cloud by using the
            PLANE_RANSAC results

        Args:
            pose_ang_lo(type: float)=np.pi/6 (default)
            pose_ang_hi(type: float)=np.pi/6*5 (default)
                -- the lower & upper bound of the angle offset between the
                -- ground plane normal vector and self.vec2sky, in radian unit

        Returns:
            None

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_valid_positive_float_bound(
            pose_ang_lo, pose_ang_hi, nameof(pose_ang_lo), nameof(pose_ang_hi)
        )
        ut.sck.is_valid_color_vector3(
            self.__grd_plane_color, "self.__grd_plane_color"
        )

        selected_index: list[int]
        if (
            self.ransac_plane_oboxes is not None
            and len(self.ransac_plane_oboxes) > 0
        ):
            num_candidates = len(self.ransac_plane_oboxes)
            selected_index = [False] * num_candidates
            vec2sky_arr = np.tile(self.vec2sky, (num_candidates, 1))
            ctrs, _, _, _, uu, vv, ww = ut.get_oriented_bounding_boxes_info(
                oboxes=self.ransac_plane_oboxes
            )
            # get hight of the obox centers along vec2sky
            ctr_hgts = np.einsum("ij,ij->i", ctrs, vec2sky_arr)
            ang_ww2sky = ut.vector_angle(ww, vec2sky_arr)
            selected_index = np.array(
                (ang_ww2sky < pose_ang_lo) | (ang_ww2sky > pose_ang_hi)
            ).tolist()

        # the ground plane is either detected with certainty or
        # constructed with minimal fail-safe method
        if selected_index is not None and np.any(np.array(selected_index)):
            # find the lowest height plane that points to self.vec2sky
            j3 = np.argmin(ctr_hgts[selected_index])
            nn = int(np.where(ctr_hgts == ctr_hgts[selected_index][j3])[0])

            # construct a big enough grd_plane
            grd_plane_model = np.array(self.ransac_plane_models[nn])
            ss = np.sign(np.dot(grd_plane_model[0:3], self.vec2sky))
            ss = 1.0 if ss == 0.0 else ss
            grd_plane_model *= np.array([ss, ss, ss, 1])

            pcd_aabox = self.pcd.get_axis_aligned_bounding_box()
            pcd_ctr: FloatArray3 = pcd_aabox.get_center()
            [ex, ey, ez] = pcd_aabox.get_extent()
            tx2sky: float = np.dot(ut.xaxis, self.vec2sky)
            ty2sky: float = np.dot(ut.yaxis, self.vec2sky)
            tz2sky: float = np.dot(ut.zaxis, self.vec2sky)
            if (np.abs(tz2sky) >= np.abs(tx2sky)) and (
                np.abs(tz2sky) >= np.abs(ty2sky)
            ):
                dist_shift = ez / 2
                lu, lv, lw = ex / 2, ey / 2, np.min([0.005, ez / 2])
            elif (np.abs(ty2sky) >= np.abs(tx2sky)) and (
                np.abs(ty2sky) >= np.abs(tz2sky)
            ):
                dist_shift = ey / 2
                lu, lv, lw = ez / 2, ex / 2, np.min([0.005, ey / 2])
            elif (np.abs(tx2sky) >= np.abs(ty2sky)) and (
                np.abs(tx2sky) >= np.abs(tz2sky)
            ):
                dist_shift = ex / 2
                lu, lv, lw = ey / 2, ez / 2, np.min([0.005, ex / 2])
            else:
                grd_plane_model = np.array(
                    [
                        self.vec2sky[0],
                        self.vec2sky[1],
                        self.vec2sky[2],
                        0,
                    ]
                )
                dist_shift = -np.dot(
                    np.array(grd_plane_model[0:3]),
                    np.array([ex / 2, ey / 2, ez / 2]),
                )
                lu, lv, lw = (
                    np.abs(dist_shift) / 2,
                    np.abs(dist_shift) / 2,
                    np.min([0.005, np.abs(dist_shift) / 2]),
                )

            # construct the 8 cubiod points for the grd_plane_obox,
            # follow outward surface normal convention
            grd_plane_ctr = pcd_ctr - grd_plane_model[0:3] * dist_shift
            grd_plane_model[3] = -np.sum(grd_plane_model[0:3] * grd_plane_ctr)
            uu, vv, ww = (
                ut.vec_norm2plane(self.vec2snr, grd_plane_model[0:3]),
                self.vec2snr,
                grd_plane_model[0:3],
            )
            pts8 = ut.calc_cuboid_vert8(grd_plane_ctr, uu, vv, ww, lu, lv, lw)
            grd_pcd = o3d.geometry.PointCloud()
            grd_pcd.points = o3d.utility.Vector3dVector(pts8)
            grd_plane_obox = (
                o3d.geometry.OrientedBoundingBox().create_from_points(
                    points=o3d.utility.Vector3dVector(pts8), robust=True
                )
            )
            self.grd_plane_obox, _, _ = ut.flip_obox(
                grd_plane_obox, self.vec2sky
            )
            self.grd_plane_obox.color = self.__grd_plane_color
            self.grd_plane_mesh = (
                o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                    grd_plane_obox, scale=[1.0, 1.0, 0.0001]
                )
            )
            self.grd_plane_mesh.paint_uniform_color(self.__grd_plane_color)
            self.grd_plane_model = grd_plane_model
            self.grd_plane_segment = self.ransac_plane_segments[nn]

            idx = self.ransac_plane_oboxes[
                nn
            ].get_point_indices_within_bounding_box(self.pcd.points)
            if idx is not None and len(idx) > 0:
                self.set_pcd_idx_label(idx, self.__grd_pcd_idx_label)
                self.set_pcd_color(idx, self.__grd_plane_color)
            else:
                grd_pcd = o3d.geometry.PointCloud()
                grd_pcd.points = o3d.utility.Vector3dVector(pts8)
                grd_pcd.paint_uniform_color(self.__grd_plane_color)
                self.grd_plane_segment = grd_pcd

            if self.disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"ground plane detected with eq: "
                    f"{self.grd_plane_model[0]:+.2f}x "
                    f"{self.grd_plane_model[1]:+.2f}y "
                    f"{self.grd_plane_model[2]:+.2f}z "
                    f"{self.grd_plane_model[3]:+.2f} = 0"
                )

            # if self.disp_info:
            #     ut.draw_planar_patches(
            #         oboxes=np.array(self.ransac_plane_oboxes)[
            #             selected_index
            #         ].tolist(),
            #         o3dobj=self.pcd,
            #         uvw_scale=0.3,
            #         uvw_selected=[True, True, True],
            #         mu_color=ut.CP.RED_DARK,
            #         mv_color=ut.CP.GREEN_DARK,
            #         mw_color=ut.CP.BLUE_DARK,
            #         disp_info=True,
            #     )

        else:
            # construct the grd_plane if needed, as fail-safe
            (
                self.grd_plane_model,
                self.grd_plane_obox,
            ) = self.construct_grd_plane_obox(ang_tol=np.pi / 36, er=0.01)
            self.grd_plane_obox.color = self.__grd_plane_color
            idx = self.grd_plane_obox.get_point_indices_within_bounding_box(
                self.pcd.points
            )
            if idx is not None and len(idx) > 0:
                self.grd_plane_segment = self.pcd.select_by_index(idx)
                self.set_pcd_idx_label(idx, self.__grd_pcd_idx_label)
                self.set_pcd_color(idx, self.__grd_plane_color)
            else:
                pts8 = self.grd_plane_obox.get_box_points()
                grd_pcd = o3d.geometry.PointCloud()
                grd_pcd.points = o3d.utility.Vector3dVector(pts8)
                grd_pcd.paint_uniform_color(self.__grd_plane_color)
                self.grd_plane_segment = grd_pcd

            # generate the grd_plane_mesh, assign colors for visualization
            self.grd_plane_mesh = (
                o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                    self.grd_plane_obox, scale=[1.0, 1.0, 0.0001]
                )
            )
            self.grd_plane_mesh.paint_uniform_color(self.__grd_plane_color)

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"ground plane constructed with eq: "
                f"{self.grd_plane_model[0]:+.2f}x "
                f"{self.grd_plane_model[1]:+.2f}y "
                f"{self.grd_plane_model[2]:+.2f}z "
                f"{self.grd_plane_model[3]:+.2f} = 0"
            )

        # if self.disp_info:
        #     _, grd_plane_obox_uvw = ut.get_o3dobj_obox_and_uvw(
        #         o3dobj=self.grd_plane_obox,
        #         obox_color=self.__grd_plane_color,
        #         uvw_scale=self.__uvw_scale,
        #     )
        #     o3d.visualization.draw(
        #         [
        #             {"name": "pcd", "geometry": self.pcd},
        #             {"name": "grd_pcd", "geometry": self.grd_plane_segment},
        #             {"name": "grd_plane_obox", "geometry": self.grd_plane_obox},
        #             {
        #                 "name": "grd_plane_obox_uvw",
        #                 "geometry": grd_plane_obox_uvw,
        #             },
        #             {"name": "grd_mesh", "geometry": self.grd_plane_mesh},
        #         ],
        #         show_ui=True,
        #     )

    def construct_grd_plane_obox(
        self,
        ang_tol: float = np.pi / 36,
        er: float = 0.02,
    ) -> Tuple[np.ndarray, OrientedBoundingBox]:
        """
        Construct a ground plane in the rare cases that the PCD-based ground
        plane detection failed.

        Args:
            ang_tol(type: float)=np.pi/36 (default) -- angle shift tolerance
                between self.vec2sky and x-, y-, z-axis, in radian unit
            er(type: float)=0.02 (default) -- the small ratio of x-, y-,
                z-extent for shifting the ground plane obox boundary planes
                to capture 3D points

        Returns:
            grd_plane_model(type: np.ndarray[1, 4]) -- constructed ground plane
            grd_plane_obox(type: o3d.geometry.OrientedBoundingBox) -- the
                oriented bounding box of the constructed ground plane

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_float_value_bounded(ang_tol, 0.0, np.pi / 12, nameof(ang_tol))
        ut.sck.is_float_value_bounded(er, 0.001, 0.1, nameof(er))

        # get the self.pcd's max & min bound, extent, diagnoal length, center
        mx = self.pcd.get_max_bound()
        mn = self.pcd.get_min_bound()
        [ex, ey, ez] = mx - mn
        pcd_ctr = self.pcd.get_center()

        # set grd_plane_model according to self.vec2sky, x,y,z-axis config
        if ut.vector_angle(self.vec2sky, ut.xaxis) <= ang_tol:
            uu, vv, _ = ut.yaxis, ut.zaxis, ut.xaxis
            grd_plane_model = ut.yzplane + np.asarray([0, 0, 0, mn[0]])
            obox_ctr = pcd_ctr + ut.xaxis * (mn[0] - pcd_ctr[0] + ex * er)
            lu, lv, lw = ey / 2, ez / 2, ex * er
        elif ut.vector_angle(self.vec2sky, ut.xaxis) >= np.pi - ang_tol:
            uu, vv, _ = ut.yaxis, ut.zaxis, ut.xaxis
            grd_plane_model = np.asarray([0, 0, 0, mx[0]]) - ut.yzplane
            obox_ctr = pcd_ctr + ut.xaxis * (mx[0] - pcd_ctr[0] - ex * er)
            lu, lv, lw = ey / 2, ez / 2, ex * er
        elif ut.vector_angle(self.vec2sky, ut.yaxis) <= ang_tol:
            uu, vv, _ = ut.zaxis, ut.xaxis, ut.yaxis
            grd_plane_model = ut.zxplane + np.asarray([0, 0, 0, mn[1]])
            obox_ctr = pcd_ctr + ut.yaxis * (mn[1] - pcd_ctr[1] + ey * er)
            lu, lv, lw = ez / 2, ex / 2, ey * er
        elif ut.vector_angle(self.vec2sky, ut.yaxis) >= np.pi - ang_tol:
            uu, vv, _ = ut.zaxis, ut.xaxis, ut.yaxis
            grd_plane_model = np.asarray([0, 0, 0, mx[1]]) - ut.zxplane
            obox_ctr = pcd_ctr + ut.yaxis * (mx[1] - pcd_ctr[1] - ey * er)
            lu, lv, lw = ez / 2, ex / 2, ey * er
        elif ut.vector_angle(self.vec2sky, ut.zaxis) <= ang_tol:
            uu, vv, _ = ut.xaxis, ut.yaxis, ut.zaxis
            grd_plane_model = ut.xyplane + np.asarray([0, 0, 0, mn[2]])
            obox_ctr = pcd_ctr + ut.zaxis * (mn[2] - pcd_ctr[2] + ez * er)
            lu, lv, lw = ex / 2, ey / 2, ez * er
        elif ut.vector_angle(self.vec2sky, ut.zaxis) >= np.pi - ang_tol:
            uu, vv, _ = ut.xaxis, ut.yaxis, ut.zaxis
            grd_plane_model = np.asarray([0, 0, 0, mx[1]]) - ut.xyplane
            obox_ctr = pcd_ctr + ut.zaxis * (mx[2] - pcd_ctr[2] - ez * er)
            lu, lv, lw = ex / 2, ey / 2, ez * er
        else:
            uu, vv, _ = ut.xaxis, ut.yaxis, ut.zaxis
            grd_plane_model = ut.xyplane
            obox_ctr = pcd_ctr
            lu, lv, lw = ex / 2, ey / 2, ez * er

        # construct the 8 cubiod points for the grd_plane_obox,
        # follow outward surface normal convention
        pts8 = ut.calc_cuboid_vert8(
            obox_ctr, uu, vv, np.asarray(self.vec2sky) * lw, lu, lv, lw
        )
        grd_plane_obox = o3d.geometry.OrientedBoundingBox().create_from_points(
            points=o3d.utility.Vector3dVector(pts8), robust=True
        )

        return (grd_plane_model, grd_plane_obox)

    def get_ref_plane(
        self,
        npt_ransac=3,
        distance_threshold=0.02,
        num_ransac_iters=500,
        pose_ang_lo=np.pi / 6,
        pose_ang_hi=np.pi / 6 * 5,
        ratio_uv_lo=0.33,
        ratio_uv_hi=3.0,
    ):
        """
        Conduct RANSAC_DBSCAN clustering with cluster obox criteria
            to detect reference plane for log depth measurements

        Args:
            npt_ransac(type: int)=3 (default)
            distance_threshold(type: float)=0.02 (default)
            num_ransac_iters(type: int)=500 (default)
            pose_ang_lo(type: float)=np.pi/6 (default)
            pose_ang_hi(type: float)=np.pi/6*5 (default)
                -- the lower & upper bound of the angle offset between the
                -- reference plane normal vector and self.vec2snr,
                -- in radian unit
            ratio_uv_lo(type: float)=0.33 (default)
            ratio_uv_hi(type: float)=3.0 (default)
                -- the lower & upper bound of the uv-plane aspect ratio
                -- for selecting the reference plane candidates, positive float

        Returns:
            N/A

        """
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=self.planar_patch_oboxes,
            o3dobjs_name="self.planar_patch_oboxes",
            selected_idx=[True] * len(self.planar_patch_oboxes),
        )
        ut.sck.is_valid_plane_model(
            plane_model=self.snr_plane_model,
            plane_model_name="self.snr_planar_model",
        )

        # select upright rod-shaped fiducials based on plane clusters'
        # poses & aspect ratios
        obox_candidates = self.planar_patch_oboxes
        num_candidates = len(obox_candidates)
        ctrs, eu, ev, ew, uu, vv, ww = ut.get_oriented_bounding_boxes_info(
            oboxes=obox_candidates
        )
        selected_index = np.repeat(False, num_candidates)
        vec2sky_arr = np.tile(self.vec2sky, (num_candidates, 1))
        vec2snr_arr = np.tile(self.vec2snr, (num_candidates, 1))
        ang_uu2sky = ut.vector_angle(uu, vec2sky_arr)
        ang_vv2sky = ut.vector_angle(vv, vec2sky_arr)
        ang_ww2sky = ut.vector_angle(ww, vec2sky_arr)
        ang_ww2sensor = ut.vector_angle(ww, vec2snr_arr)
        selected_index = (
            ((ang_ww2sensor < pose_ang_lo) | (ang_ww2sensor > pose_ang_hi))
            & ((ang_ww2sky > np.pi / 4) & (ang_ww2sky < np.pi - np.pi / 4))
            & (
                (
                    (ev / eu >= ratio_uv_hi)
                    & (
                        (ang_vv2sky <= pose_ang_lo)
                        | (ang_vv2sky >= pose_ang_hi)
                    )
                )
                | (
                    (ev / eu <= ratio_uv_lo)
                    & (
                        (ang_uu2sky <= pose_ang_lo)
                        | (ang_uu2sky >= pose_ang_hi)
                    )
                )
            )
        )

        # calc the distance between the patch centers and the sensor plane
        aa = np.hstack((ctrs, np.ones((len(ctrs), 1))))
        bb = np.repeat(
            np.asarray(self.snr_plane_model)[:, np.newaxis], len(ctrs), 1
        ).T
        dist_arr = np.abs(
            np.sum(aa * bb, axis=1)
            / (np.sum(bb[:, 0:3] * bb[:, 0:3], axis=1) ** 0.5)
        )

        self.fid_segments = []
        self.fid_oboxes = []
        if len(dist_arr[selected_index]) >= 2:
            sorted_index_arr = np.argsort(dist_arr[selected_index])
            sorted_arr = dist_arr[selected_index][sorted_index_arr]
            nn_arr = np.where(dist_arr[selected_index] <= sorted_arr[1])[0]
            if self.disp_info:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"The first two shortest distances are: "
                    f"{sorted_arr[0]:.2e}, {sorted_arr[1]:.2e}"
                )
            for nn in nn_arr:
                seg = np.array(self.planar_patch_segments)[selected_index][nn]
                seg.paint_uniform_color(ut.CP.RED_DARK)
                self.fid_segments.append(seg)
                obox = np.array(self.planar_patch_oboxes)[selected_index][nn]
                obox.color = ut.CP.RED
                self.fid_oboxes.append(obox)
        elif len(dist_arr[selected_index]) == 1:
            if self.disp_info:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"The shortest distance is: "
                    f"{dist_arr[selected_index][0]:.2e}"
                )
            # for nn in nn_arr:
            seg = np.array(self.planar_patch_segments)[selected_index][0]
            seg.paint_uniform_color(ut.CP.RED_DARK)
            self.fid_segments.append(seg)
            obox = np.array(self.planar_patch_oboxes)[selected_index][0]
            obox.color = ut.CP.RED
            self.fid_oboxes.append(obox)
        else:
            if self.disp_info:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"Could not find two large aspect ratio oboxes with "
                    f"the shortest distances, use the 1st available obox "
                    f"as the selections, be cautious!"
                )
            seg = np.array(self.planar_patch_segments)[0]
            seg.paint_uniform_color(ut.CP.RED_DARK)
            self.fid_segments.append(seg)
            obox = np.array(self.planar_patch_oboxes)[0]
            obox.color = ut.CP.RED
            self.fid_oboxes.append(obox)

        # construct a ref_plane based on the detected fiducial clusters
        self.ref_plane_segment, labels = ut.assemble_pcd_clusters(
            pcd_clusters=self.fid_segments,
            kdtree_radius=0.01,
            kdtree_max_nn=10,
        )
        self.ref_plane_segment.paint_uniform_color(self.__ref_plane_color)
        (
            self.ref_plane_model,
            inliers,
        ) = self.ref_plane_segment.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=npt_ransac,
            num_iterations=num_ransac_iters,
        )
        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"reference plane detected & constructed with eq: "
                f"{self.ref_plane_model[0]:+.2f}x "
                f"{self.ref_plane_model[1]:+.2f}y "
                f"{self.ref_plane_model[2]:+.2f}z "
                f"{self.ref_plane_model[3]:+.2f} = 0"
            )

        # construct a ref_plane_obox
        self.ref_plane_obox = self.ref_plane_segment.get_oriented_bounding_box(
            robust=True
        )
        self.ref_plane_obox.color = self.__ref_plane_color
        self.ref_plane_mesh = (
            o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                self.ref_plane_obox, scale=[1.2, 1.2, 0.0001]
            )
        )
        self.ref_plane_mesh.paint_uniform_color(self.__ref_plane_color)

        idx = self.ref_plane_obox.get_point_indices_within_bounding_box(
            self.pcd.points
        )
        if idx is not None and len(idx) > 0:
            self.set_pcd_idx_label(idx, self.__ref_pcd_idx_label)
            self.set_pcd_color(idx, self.__ref_plane_color)

        # show the results as debug info
        # if self.disp_info:
        #     ut.draw_planar_patches(
        #         oboxes=np.array(self.fid_oboxes).tolist(),
        #         o3dobj=self.pcd,
        #         uvw_scale=0.2,
        #         uvw_selected=np.array([True, True, True]),
        #         mu_color=ut.CP.RED_DARK,
        #         mv_color=ut.CP.GREEN_DARK,
        #         mw_color=ut.CP.BLUE_DARK,
        #         disp_info=True,
        #     )

    def get_snr_plane(self) -> None:
        """
        Detect the sensor plane in the point cloud

        Args:
            None

        Returns:
            None

        """
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_valid_color_vector3(
            self.__snr_plane_color, "self.__snr_plane_color"
        )
        ut.sck.is_valid_color_vector3(
            self.__snr_arr_color, "self.__snr_arr_color"
        )

        # pick up the camera capture positions from the sfm.ply pcd,
        # find the green pts in the first 100 pts, snoload uses 48 captures
        snr_idx: list[int] = np.where(
            (np.array(self.pcd.colors)[0:100, :] == ut.CP.GREEN).all(axis=1)
        )[0].tolist()
        snr_pcd: PointCloud
        snr_plane_model: FloatArray4
        snr_obox: OrientedBoundingBox
        snr_obox_uvw: TriangleMesh
        snr_plane_mesh: TriangleMesh
        uu: FloatArray3
        vv: FloatArray3
        ww: FloatArray3

        if snr_idx is not None and len(snr_idx) > 0:
            # if the cam capture positions are in the pcd
            snr_pcd = self.pcd.select_by_index(snr_idx)
            snr_pcd.paint_uniform_color(self.__snr_arr_color)
            self.set_pcd_idx_label(snr_idx, self.__snr_pcd_idx_label)
            self.set_pcd_color(snr_idx, self.__snr_plane_color)

            snr_plane_model, _ = snr_pcd.segment_plane(
                distance_threshold=0.002,
                ransac_n=3,
                num_iterations=100,
            )
            # make sure the snr_plane_model norm_vec points to -self.vec2snr
            ss = np.sign(np.dot(snr_plane_model[0:3], -self.vec2snr))
            ss = 1.0 if ss == 0.0 else ss
            snr_plane_model *= np.array([ss, ss, ss, 1])

            # for future improvement
            # calc self.vec2sky's projection vector on the snr_plane_model
            # vec2sky_on_snr_plane: FloatArray3 = ut.vec_proj_on_plane(
            #     self.vec2sky, snr_plane_model
            # )

            snr_obox, snr_obox_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=snr_pcd,
                obox_color=self.__snr_plane_color,
                uvw_scale=self.__uvw_scale,
                uvw_selected=np.array([True, True, True]),
                mu_color=ut.CP.RED,
                mv_color=ut.CP.GREEN,
                mw_color=ut.CP.BLUE,
            )

            # for future improvement
            # rotate the obox along its ww-axis to align its upright vector
            # (either uu or vv depending on which one is closer to vec2sky's
            #  projection vector on the obox_plane), so the obox aligns better
            #  with the tri-axes
            # snr_obox_t, rotmat, octr = ut.spin_oriented_bounding_box(
            #     copy.deepcopy(snr_obox), vec2sky_on_snr_plane
            # )
            # snr_uvw_t: TriangleMesh = (
            #     copy.deepcopy(snr_obox_uvw)
            #     .translate(translation=-octr, relative=True)
            #     .rotate(R=rotmat, center=(0, 0, 0))
            #     .translate(translation=octr, relative=True)
            # )

            # flip the snr_obox if its ww points to self.vec2snr,  so the
            # final sensor plane obox ww points to the log
            snr_plane_obox, rotmat, octr = ut.flip_obox(
                copy.deepcopy(snr_obox), -self.vec2snr
            )
            snr_plane_obox.color = self.__snr_plane_color
            snr_plane_obox_uvw = (
                copy.deepcopy(snr_obox_uvw)
                .translate(translation=-octr, relative=True)
                .rotate(R=rotmat, center=(0, 0, 0))
                .translate(translation=octr, relative=True)
            )
            snr_plane_mesh = (
                o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                    snr_plane_obox, scale=[1.0, 1.0, 0.0001]
                )
            )
            snr_plane_mesh.paint_uniform_color(self.__snr_plane_color)

        else:
            # else, create snr_plane_obox
            pcd_aabox = self.pcd.get_axis_aligned_bounding_box()
            pcd_ctr: FloatArray3 = pcd_aabox.get_center()
            [ex, ey, ez] = pcd_aabox.get_extent()
            tx2sensor: float = np.dot(ut.xaxis, self.vec2snr)
            ty2sensor: float = np.dot(ut.yaxis, self.vec2snr)
            tz2sensor: float = np.dot(ut.zaxis, self.vec2snr)
            if (np.abs(tz2sensor) >= np.abs(tx2sensor)) and (
                np.abs(tz2sensor) >= np.abs(ty2sensor)
            ):
                ss = np.sign(np.dot(ut.zaxis, -self.vec2snr))
                ss = 1.0 if ss == 0.0 else ss
                snr_plane_model = ut.xyplane * ss
                snr_plane_ctr = pcd_ctr - snr_plane_model[0:3] * (ez / 2)
                snr_plane_model[3] = snr_plane_ctr[2]
                uu, vv, ww = ss * ut.xaxis, ss * ut.yaxis, snr_plane_model[0:3]
                lu, lv, lw = ex / 2, ey / 2, np.min([0.005, ez / 2])
            elif (np.abs(ty2sensor) >= np.abs(tx2sensor)) and (
                np.abs(ty2sensor) >= np.abs(tz2sensor)
            ):
                ss = np.sign(np.dot(ut.yaxis, -self.vec2snr))
                ss = 1.0 if ss == 0.0 else ss
                snr_plane_model = ut.zxplane * ss
                snr_plane_ctr = pcd_ctr - snr_plane_model[0:3] * (ey / 2)
                snr_plane_model[3] = snr_plane_ctr[1]
                uu, vv, ww = ss * ut.zaxis, ss * ut.xaxis, snr_plane_model[0:3]
                lu, lv, lw = ez / 2, ex / 2, np.min([0.005, ey / 2])
            elif (np.abs(tx2sensor) >= np.abs(ty2sensor)) and (
                np.abs(tx2sensor) >= np.abs(tz2sensor)
            ):
                ss = np.sign(np.dot(ut.xaxis, -self.vec2snr))
                ss = 1.0 if ss == 0.0 else ss
                snr_plane_model = ut.yzplane * ss
                snr_plane_ctr = pcd_ctr - snr_plane_model[0:3] * (ex / 2)
                snr_plane_model[3] = snr_plane_ctr[0]
                uu, vv, ww = ss * ut.yaxis, ss * ut.zaxis, snr_plane_model[0:3]
                lu, lv, lw = ey / 2, ez / 2, np.min([0.005, ex / 2])
            else:
                snr_plane_model = np.array(
                    [
                        -self.vec2snr[0],
                        -self.vec2snr[1],
                        -self.vec2snr[2],
                        0,
                    ]
                )
                dd = np.dot(snr_plane_model[0:3], [ex / 2, ey / 2, ez / 2])
                snr_plane_ctr = pcd_ctr - snr_plane_model[0:3] * dd
                snr_plane_model[3] = -np.sum(
                    snr_plane_model[0:3] * snr_plane_ctr
                )
                uu, vv, ww = (
                    ut.vec_norm2plane(self.vec2sky, snr_plane_model[0:3]),
                    self.vec2sky,
                    snr_plane_model[0:3],
                )
                lu, lv, lw = np.abs(dd) / 2, np.abs(dd) / 2, np.abs(dd) / 2
                lw = np.min([0.005, lw])

            # construct the 8 cubiod points for the snr_plane_obox,
            # follow outward surface normal convention
            pts8 = ut.calc_cuboid_vert8(snr_plane_ctr, uu, vv, ww, lu, lv, lw)
            snr_plane_obox = (
                o3d.geometry.OrientedBoundingBox().create_from_points(
                    points=o3d.utility.Vector3dVector(pts8), robust=True
                )
            )
            _, snr_plane_obox_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=snr_plane_obox,
                obox_color=self.__snr_plane_color,
                uvw_scale=self.__uvw_scale,
            )
            snr_plane_mesh = (
                o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                    snr_plane_obox, scale=[1.0, 1.0, 0.0001]
                )
            )
            snr_plane_mesh.paint_uniform_color(self.__snr_plane_color)
            idx = snr_plane_obox.get_point_indices_within_bounding_box(
                self.pcd.points
            )
            if idx is not None and len(idx) > 0:
                self.set_pcd_idx_label(idx, self.__snr_pcd_idx_label)
                self.set_pcd_color(idx, self.__snr_plane_color)
            else:
                snr_pcd = o3d.geometry.PointCloud()
                snr_pcd.points = o3d.utility.Vector3dVector(pts8)
                snr_pcd.paint_uniform_color(self.__snr_plane_color)

        # sensor plane parameters
        self.snr_plane_model = snr_plane_model
        # self.snr_plane_segment = snr_pcd
        self.snr_plane_obox = snr_plane_obox
        self.snr_plane_mesh = snr_plane_mesh
        # snr_arr_mesh: TriangleMesh = o3d.geometry.TriangleMesh()
        # for jj, pt in enumerate(snr_pcd.points):
        #     snr_arr_mesh = snr_arr_mesh + ut.get_arrow_mesh(
        #         origin=pt,
        #         vector=pt + snr_plane_obox.R[:, 2],
        #         scale=0.1,
        #         arrow_color=self.__snr_arr_color,
        #         cylinder_radius=0.25,
        #         cylinder_height=1.0,
        #         cylinder_split=4,
        #         cone_radius=0.35,
        #         cone_height=0.5,
        #         cone_split=1,
        #         resolution=20,
        #     )
        # self.snr_arr_mesh = snr_arr_mesh

        # if self.disp_info:
        #     o3d.visualization.draw(
        #         [
        #             {"name": "pcd", "geometry": self.pcd},
        #             {"name": "snr_pcd", "geometry": self.snr_plane_segment},
        #             {"name": "snr_arr", "geometry": self.snr_arr_mesh},
        #             {"name": "snr_obox", "geometry": snr_obox},
        #             {"name": "snr_obox_uvw", "geometry": snr_obox_uvw},
        #             # {"name": "snr_obox_t", "geometry": snr_obox_t},
        #             # {"name": "snr_obox_t_uvw", "geometry": snr_uvw_t},
        #             {"name": "snr_plane_obox", "geometry": self.snr_plane_obox},
        #             {
        #                 "name": "snr_plane_obox_uvw",
        #                 "geometry": snr_plane_obox_uvw,
        #             },
        #             {"name": "snr_mesh", "geometry": self.snr_plane_mesh},
        #         ],
        #         show_ui=True,
        #         # point_size=5,
        #     )

    def filter_planar_patches(
        self,
        obox_candidates: list[OrientedBoundingBox],
        pose_ang_lo: float = 45.0,
        pose_ang_hi: float = 135.0,
        diag_uv_lo: float = 0.15,
        diag_uv_hi: float = 0.9,
        ratio_uv_lo: float = 0.5,
        ratio_uv_hi: float = 2.0,
        grd_hgt_lo: float = 0.3,
    ) -> None:
        """
        Filter multiple oriented bounding boxes based on the given conditions:
        obox's uv plane diagonal, aspect ratio, and uv plane normal vector
        alignment angle with the given ref_vec, the distance to ground plane.

        Args:
            pose_ang_lo=np.pi/4 (default): pose vec alignment lo bound angle
            pose_ang_hi=np.pi/4*3 (default): pose vec alignment hi bound angle
            diag_uv_lo=0.15 (default) -- obox's uv plane diagonal lo bound
            diag_uv_hi=0.90 (default) -- obox's uv plane diagonal hi bound
            ratio_uv_lo=0.5 (default) -- obox's uv plane aspect ratio lo bound
            ratio_uv_hi=2.0 (default) -- obox's uv plane aspect ratio hi bound
            grd_hght_lo=0.3 (default) -- obox's ctr ground clearance lo bound
            disp_info=True (default) -- controls info display

        Returns:
            None

        """
        # obox_candidates = self.planar_patch_oboxes
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=obox_candidates,
            o3dobjs_name="obox_candidates",
            selected_idx=[True] * len(obox_candidates),
        )
        ref_vec = self.vec2snr
        grd_plane_model = self.grd_plane_model

        # Sanity check for the input arguments
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=obox_candidates,
            o3dobjs_name=nameof(obox_candidates),
            selected_idx=[True] * len(obox_candidates),
        )
        ut.sck.is_nonzero_vector3(ref_vec, nameof(ref_vec))
        ut.sck.is_valid_positive_float_bound(
            pose_ang_lo, pose_ang_hi, nameof(pose_ang_lo), nameof(pose_ang_hi)
        )
        ut.sck.is_valid_positive_float_bound(
            diag_uv_lo, diag_uv_hi, nameof(diag_uv_lo), nameof(diag_uv_hi)
        )
        ut.sck.is_valid_positive_float_bound(
            ratio_uv_lo, ratio_uv_hi, nameof(ratio_uv_lo), nameof(ratio_uv_hi)
        )
        ut.sck.is_positive_float(grd_hgt_lo, "grd_hgt_lo")
        ut.sck.is_valid_plane_model(grd_plane_model, "grd_plane_model")

        # collect necessary metrics for the screening in later step
        ref_vec = ref_vec / np.linalg.norm(ref_vec + 1e-16)
        [aa, bb, cc, dd] = grd_plane_model
        num_candidates = len(obox_candidates)
        selected_index = [False] * num_candidates
        ctrs, eu, ev, ew, uu, vv, ww = ut.get_oriented_bounding_boxes_info(
            oboxes=obox_candidates
        )
        eduv = (eu**2 + ev**2) ** 0.5
        asuv = np.maximum(eu / ev, ev / eu)
        ang_ww2refvec = np.array(
            ut.vector_angle(ww, np.tile(ref_vec, (ww.shape[0], 1)))
        )
        grd_hgt = np.array(
            np.abs(aa * ctrs[:, 0] + bb * ctrs[:, 1] + cc * ctrs[:, 2] + dd)
            / (aa * aa + bb * bb + cc * cc) ** 0.5
        )

        # screen the obox_candidates
        selected_index = (
            ((ang_ww2refvec <= pose_ang_lo) | (ang_ww2refvec >= pose_ang_hi))
            & ((eduv >= diag_uv_lo) & (eduv <= diag_uv_hi))
            & ((asuv >= ratio_uv_lo) & (asuv <= ratio_uv_hi))
            & (grd_hgt >= grd_hgt_lo)
        )
        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: planar patches -- patch#, "
                f"eu, ev, eduv, asuv, ang_ww2refvec, grd-hgt, selected"
            )
            [
                print(
                    f"  #{jj:02d}: {eu[jj]:.2e} | {ev[jj]:.2e} | "
                    f"{eduv[jj]:.2e} | {asuv[jj]:.2e} | "
                    f"{int(ang_ww2refvec[jj] / np.pi *180):03d} | "
                    f"{grd_hgt[jj]:.2e} | {selected_index[jj]}"
                )
                for jj in range(num_candidates)
            ]

        # set relevant results
        self.planar_patch_selected_idx = (list)(selected_index)

    def get_log_scaling_results(self) -> None:
        """
        Get log scaling result for the given poing cloud (at one end)

        Args:
            None

        Returns:
            None

        """
        # Sanity check
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=self.planar_patch_oboxes,
            o3dobjs_name=nameof(self.planar_patch_oboxes),
            selected_idx=self.planar_patch_selected_idx,
        )
        ut.sck.is_valid_plane_model(
            plane_model=self.ref_plane_model,
            plane_model_name="ref_plane_model",
        )

        oboxes = np.asarray(self.planar_patch_oboxes)[
            self.planar_patch_selected_idx
        ]
        # collect necessary metrics
        ctrs, eu, ev, ew, uu, vv, ww = ut.get_oriented_bounding_boxes_info(
            oboxes=oboxes.tolist()
        )
        dmx = np.maximum(eu, ev)
        dmn = np.minimum(eu, ev)
        aa = np.hstack((ctrs, np.ones((len(ctrs), 1))))
        bb = np.repeat(
            np.asarray(self.ref_plane_model)[:, np.newaxis], len(ctrs), 1
        ).T
        dist2ref = (
            -np.sum(aa * bb, axis=1)
            / (np.sum(bb[:, 0:3] * bb[:, 0:3], axis=1) ** 0.5).tolist()
        )
        meshes = [
            o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                obox, scale=[1, 1, 0.0001]
            )
            for obox in oboxes
        ]
        for jj, mesh in enumerate(meshes):
            oboxes[jj].color = ut.CP.RED_DARK
            mesh.paint_uniform_color(ut.CP.CYAN)

        self.logend_diameters_long = dmx.tolist()
        self.logend_diameters_short = dmn.tolist()
        self.logend_dist2ref = dist2ref.tolist()
        self.logend_nvecs = ww.tolist()
        self.logend_centers = ctrs.tolist()
        self.logend_oboxes = oboxes.tolist()
        self.logend_segments = np.asarray(self.planar_patch_segments)[
            self.planar_patch_selected_idx
        ].tolist()
        self.logend_meshes = meshes

        if self.disp_progress:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"scaling results: log#, d-long, d-short, dist2ref, ctr, nvec"
            )
            [
                print(
                    f"  #{jj:02d}: {self.logend_diameters_long[jj]:.2e} | "
                    f"{self.logend_diameters_short[jj]:.2e} | "
                    f"{self.logend_dist2ref[jj]:+.2e} | "
                    f"[{self.logend_centers[jj][0]:+.2e}, "
                    f"{self.logend_centers[jj][1]:+.2e}, "
                    f"{self.logend_centers[jj][2]:+.2e}] | "
                    f"[{self.logend_nvecs[jj][0]:+.2e}, "
                    f"{self.logend_nvecs[jj][1]:+.2e}, "
                    f"{self.logend_nvecs[jj][2]:+.2e}]"
                )
                for jj, obox in enumerate(self.logend_oboxes)
            ]

    def draw_log_scaling_results(
        self,
        vis: o3d.pybind.visualization.O3DVisualizer,
        label_name: str,
        uvw_scale: float = 1.0,
        uvw_selected: BoolArray3 = np.array([True, True, True]),
        mu_color: np.ndarray = ut.CP.RED_DARK,
        mv_color: np.ndarray = ut.CP.GREEN_DARK,
        mw_color: np.ndarray = ut.CP.BLUE_DARK,
    ) -> None:
        """
        Draw planar patches with optional overlay on the given point cloud

        Args:
            vis(type: o3d.visualization.O3DVisualizer)
            label_name(type: str)
            uvw_scale(type: float)=1.0 (default)
                -- set uvw arrow mesh scale
            uvw_selected(type: list[bool])=[True, True, True] (default)
                -- control which uvw arrow mesh to display
            mu_color(type: np.array([r, g, b]))=CP.RED_DARK (defalut)
            mv_color(type: np.array([r, g, b]))=CP.GREEN_DARK (defalut)
            mw_color(type: np.array([r, g, b]))=CP.BLUE_DARK (defalut)

        Returns:
            None

        """
        # sanity check for the relevant results within the class instance
        ut.sck.is_valid_pcd(pcd=self.pcd, pcd_name="pcd")
        ut.sck.is_valid_plane_model(
            plane_model=self.grd_plane_model,
            plane_model_name="grd_plane_model",
        )
        ut.sck.is_valid_plane_model(
            plane_model=self.ref_plane_model,
            plane_model_name="ref_plane_model",
        )
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=self.logend_oboxes,
            o3dobjs_name=nameof(self.logend_oboxes),
            selected_idx=[True] * len(self.logend_oboxes),
        )
        ut.sck.is_valid_o3dobj_list(
            o3dobjs=self.fid_oboxes,
            o3dobjs_name=nameof(self.fid_oboxes),
            selected_idx=[True] * len(self.fid_oboxes),
        )

        # sanity check on the input arguments
        ut.sck.is_positive_float(uvw_scale, nameof(uvw_scale))
        ut.sck.is_valid_color_vector3(mu_color, nameof(mu_color))
        ut.sck.is_valid_color_vector3(mv_color, nameof(mv_color))
        ut.sck.is_valid_color_vector3(mw_color, nameof(mw_color))

        meshes = self.logend_meshes
        oboxes = self.logend_oboxes
        uu = np.array([obox.R[:, 0] for obox in oboxes])
        vv = np.array([obox.R[:, 1] for obox in oboxes])
        ww = self.logend_nvecs
        ctrs = self.logend_centers
        dmx = self.logend_diameters_long
        dmn = self.logend_diameters_short
        dist2ref = self.logend_dist2ref

        dmx_en = np.round(np.array(dmx) * ut.m2mm / ut.inch2mm * 8) / 8
        dmn_en = np.round(np.array(dmn) * ut.m2mm / ut.inch2mm * 8) / 8
        dist2ref_en = (
            np.round(np.array(dist2ref) * ut.m2mm / ut.inch2mm * 8) / 8
        )

        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"scaling results: log#, d1, d2, d_avg, dist2ref"
            )

        vis.add_geometry(f"{label_name}_pcd", self.pcd)
        vis.add_geometry(f"{label_name}_grd", self.grd_plane_mesh)
        vis.add_geometry(f"{label_name}_ref", self.ref_plane_mesh)
        vis.add_geometry(f"{label_name}_snr", self.snr_plane_mesh)
        for jj, fid_obox in enumerate(self.fid_oboxes):
            vis.add_geometry(f"{label_name}_fid{jj:02d}", fid_obox)
        # add the visualization elements
        for jj, obox in enumerate(oboxes):
            vis.add_geometry(f"{label_name}_obox{jj:02d}", obox)
            ctr = np.array(ctrs[jj])
            nu, nv, nw = np.array(uu[jj]), np.array(vv[jj]), np.array(ww[jj])
            if np.any(uvw_selected):
                mu = ut.get_arrow_mesh(ctr, ctr + nu, uvw_scale, mu_color)
                mv = ut.get_arrow_mesh(ctr, ctr + nv, uvw_scale, mv_color)
                mw = ut.get_arrow_mesh(ctr, ctr + nw, uvw_scale, mw_color)
                uvw_mesh = np.sum(
                    np.array([mu, mv, mw])[np.array(uvw_selected)]
                )
                vis.add_geometry(
                    f"{label_name}_mesh{jj:02d}", meshes[jj] + uvw_mesh
                )
            else:
                vis.add_geometry(f"{label_name}_mesh{jj:02d}", meshes[jj])
            vis.add_3d_label(
                ctr,
                f"#{jj}|{int(np.round((dmx_en[jj] + dmn_en[jj])/2))}|"
                f"{dist2ref_en[jj]:.3f}\n"
                f"{dmx_en[jj]:.3f}\n{dmn_en[jj]:.3f}"
                # f"{jj}|({int(dmn[jj] * ut.m2mm)}|{int(dmx[jj] * ut.m2mm)}|"
                # f"{int(dist2ref[jj] * ut.m2mm)})",
            )

            if self.disp_info:
                print(
                    f"{jj:02d}|({dmx_en[jj]:6.3f} | {dmn_en[jj]:6.3f} | "
                    f"{int(np.round((dmx_en[jj] + dmn_en[jj])/2)):02d} | "
                    f"{dist2ref_en[jj]:+7.3f})"
                )
                # print(
                #     f"  #{jj:02d}: {dmx[jj]:.2e} | {dmn[jj]:.2e} | "
                #     f"{dist2ref[jj]:+.2e} | [{ctrs[jj][0]:+.2e}, "
                #     f"{ctrs[jj][1]:+.2e}, {ctrs[jj][2]:+.2e}] | "
                #     f"[{ww[jj][0]:+.2e}, {ww[jj][1]:+.2e}, {ww[jj][2]:+.2e}]"
                # )

    def preprocess_pcd(
        self,
        dst_pcd_axes: FloatArray3x3 = np.eye(3),
        ang_offset: float = np.pi / 6,
        max_num_planes: int = 50,
    ) -> PointCloud:
        """
        preprocess PCD to align it with xyz axes

        Args:
            dst_pcd_axes(type: np.array)=nd.array([xaxis, yaxis, zaxis]).T
                -- the desired three orthogonal axes for the processed PCD
            ang_offset(type: float)=np.pi/6 (default)
                -- angle offset tolerance, in radian unit
            max_num_planes(type: int)=50 (default)
                -- max number of planes for plane_ransac segmentation

        Returns:
            pcd_dst (PointCloud)
                -- the preprocessed PCD

        """
        # Sanity check for the input arguments
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_float(ang_offset, nameof(ang_offset))

        # run plane_ransac to find the plane PCD clusters
        self.plane_ransac(
            npt_ransac=3,
            distance_threshold=0.01,
            num_ransac_iters=500,
            max_num_planes=max_num_planes,
        )

        # get the ground plane
        self.get_grd_plane(
            pose_ang_lo=ang_offset,
            pose_ang_hi=np.pi - ang_offset,
        )
        # update vec2sky if needed
        old_ww = self.vec2sky
        new_ww = self.grd_plane_obox.R[:, 2]
        ang_off, vec_updated = ut.vector_angle(new_ww, self.vec2sky), False
        if ang_off <= ang_offset:
            self.vec2sky, vec_updated = new_ww, True
        elif ang_off >= np.pi - ang_offset:
            self.vec2sky, vec_updated = -new_ww, True
        if self.disp_progress and vec_updated:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"The pcd's vec2sky has been updated from [{old_ww[0]:+.2e}, "
                f"{old_ww[1]:+.2e}, {old_ww[2]:+.2e}] to ["
                f"{self.vec2sky[0]:+.2e}, {self.vec2sky[1]:+.2e}, "
                f"{self.vec2sky[2]:+.2e}]"
            )

        # rotate the grd_plane_obox so its ww vector points along z-axis
        self.grd_plane_obox, _, _ = ut.flip_obox(
            self.grd_plane_obox, self.vec2sky
        )
        grd_plane_obox_ctr = self.grd_plane_obox.get_center()
        src_pcd_axes = self.grd_plane_obox.R
        rotmat, trans = ut.rigid_transform_3D(src_pcd_axes, dst_pcd_axes)
        pcd_dst = (
            copy.deepcopy(self.pcd)
            .translate(translation=-grd_plane_obox_ctr, relative=True)
            .rotate(R=rotmat, center=(0, 0, 0))
            # .translate(translation=grd_plane_obox_ctr, relative=True)
        )

        if self.disp_info:
            print(
                f"[INFO: {ut.currentFuncName()}]:\n"
                f"    rotation matrix:\n"
                f"      |{rotmat[0, 0]:+.2e}, {rotmat[0, 1]:+.2e}, "
                f"{rotmat[0, 2]:+.2e}|\n"
                f"      |{rotmat[1, 0]:+.2e}, {rotmat[1, 1]:+.2e}, "
                f"{rotmat[1, 2]:+.2e}|\n"
                f"      |{rotmat[2, 0]:+.2e}, {rotmat[2, 1]:+.2e}, "
                f"{rotmat[2, 2]:+.2e}|\n"
                f"    translation vector: \n"
                f"      [{trans[0]:.2e}, {trans[1]:.2e}, {trans[2]:.2e}]"
            )

            obox, uvw_mesh = ut.get_o3dobj_obox_and_uvw(
                o3dobj=self.pcd,
                obox_color=ut.CP.BLUE,
                uvw_scale=self.__uvw_scale,
            )
            obox_t, uvw_t = ut.get_o3dobj_obox_and_uvw(
                o3dobj=pcd_dst,
                obox_color=ut.CP.CYAN,
                uvw_scale=self.__uvw_scale,
            )
            grd_obox, grd_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=self.grd_plane_segment,
                obox_color=ut.CP.ORANGE,
                uvw_scale=self.__uvw_scale,
            )
            o3d.visualization.draw(
                [
                    {"name": "src_pcd", "geometry": self.pcd},
                    {"name": "src_obox", "geometry": obox},
                    {"name": "src_obox_uvw", "geometry": uvw_mesh},
                    {"name": "grd_obox", "geometry": grd_obox},
                    {"name": "grd_obox_uvw", "geometry": grd_uvw},
                    {
                        "name": "self.grd_plane_obox",
                        "geometry": self.grd_plane_obox,
                    },
                    {"name": "dst_pcd", "geometry": pcd_dst},
                    {"name": "dst_obox", "geometry": obox_t},
                    {"name": "dst_obox_uvw", "geometry": uvw_t},
                ],
                show_ui=True,
            )

        return pcd_dst

    def logend_scaling_prep(
        self,
        param_patch_det: ut.ParamPlanarPatchDetection,
        param_fid_det: ut.ParamFiducialDetection,
        ang_offset: float = np.pi / 6,
        max_num_planes: int = 50,
        ref_plane_axis_position: float = 0.0,
    ) -> None:
        """
        Prepare for the log scaling step -- redo ground plane detection,
            detect reference- and sensor- planes, then align the reference
            plane at its desired global axis position

        Args:
            param_patch_det (Type[ut.ParamPlanarPatchDetection])
                -- param for planar patch detection
            param_fid_det (Type[ut.ParamFiducialDetection])
                -- param for fiducial detection
            ang_offset(type: float)=np.pi/6 (default)
                -- angle offset tolerance, radian unit
            max_num_planes(type: int)=50 (default)
                -- max number of planes for plane_ransac segmentation
            ref_plane_axis_position(type: float)=0.0 (default)
                -- the desired global axis position for aligning the reference plane
        Returns:
            None

        """
        # Sanity check for the input arguments
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")
        ut.sck.is_positive_float(ang_offset, nameof(ang_offset))
        ut.sck.is_positive_int(max_num_planes, nameof(max_num_planes))

        # run plane_ransac to find the plane PCD clusters
        self.plane_ransac(
            npt_ransac=3,
            distance_threshold=0.01,
            num_ransac_iters=500,
            max_num_planes=max_num_planes,
        )
        # get the ground plane
        self.get_grd_plane(
            pose_ang_lo=ang_offset,
            pose_ang_hi=np.pi - ang_offset,
        )
        # update vec2sky if needed
        old_ww = self.grd_plane_obox.R[:, 2]
        new_ww = self.grd_plane_obox.R[:, 2]
        ang_off, vec_updated = ut.vector_angle(new_ww, self.vec2sky), False
        if ang_off <= ang_offset:
            self.vec2sky, vec_updated = new_ww, True
        elif ang_off >= np.pi - ang_offset:
            self.vec2sky, vec_updated = -new_ww, True
        if self.disp_progress and vec_updated:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"The pcd's vec2sky has been updated from [{old_ww[0]:+.2e}, "
                f"{old_ww[1]:+.2e}, {old_ww[2]:+.2e}] to ["
                f"{self.vec2sky[0]:.2e}, {self.vec2sky[1]:.2e}, "
                f"{self.vec2sky[2]:.2e}]"
            )
        # get the snr plane
        self.get_snr_plane()

        # run planar patch detection to get rectangle patches
        self.planar_patches(
            the_pcd=self.pcd,
            normal_variance_threshold_deg=param_patch_det.normal_variance_threshold_deg,  # noqa
            coplanarity_deg=param_patch_det.coplanarity_deg,
            outlier_ratio=param_patch_det.outlier_ratio,
            min_plane_edge_length=param_patch_det.min_plane_edge_len,
            min_num_points=param_patch_det.min_num_pts,
            search_knn=param_patch_det.search_knn,
        )
        if self.disp_info:
            ut.draw_planar_patches(
                oboxes=self.planar_patch_oboxes,
                o3dobj=self.pcd,
                uvw_scale=0.2,
                uvw_selected=np.array([True, True, True]),
                disp_info=self.disp_info,
            )
        # get the referece plane for the actual vec2snr, and for
        # depth measurements and later proper visual
        self.get_ref_plane(
            pose_ang_lo=param_fid_det.fid_patch_ang_lo,
            pose_ang_hi=param_fid_det.fid_patch_ang_hi,
            ratio_uv_lo=param_fid_det.fid_patch_ratio_uv_lo,
            ratio_uv_hi=param_fid_det.fid_patch_ratio_uv_hi,
        )
        old_ww = self.ref_plane_obox.R[:, 2]
        new_ww = self.ref_plane_obox.R[:, 2]
        ang_off = ut.vector_angle(new_ww, self.vec2snr)
        ang2sky = ut.vector_angle(new_ww, self.vec2sky)
        vec_updated = False
        if np.abs(ang2sky - np.pi / 2) <= ang_offset:
            if ang_off <= ang_offset:
                self.vec2snr, vec_updated = new_ww, True
            elif ang_off >= np.pi - ang_offset:
                self.vec2snr, vec_updated = -new_ww, True
        if self.disp_progress and vec_updated:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"The pcd's vec2snr has been updated from "
                f"[{old_ww[0]:.2e}, {old_ww[1]:.2e}, {old_ww[2]:.2e}] to "
                f"[{self.vec2snr[0]:.2e}, {self.vec2snr[1]:.2e}, "
                f"{self.vec2snr[2]:.2e}]"
            )

        # first shift the pcd such that its reference plane is at vec2snr's
        # origin position, then move to the tvec position as defined
        # this is still a rough implementation. Will improve later.
        # sensor_axis_shift = (
        #     np.dot(
        #         self.ref_plane_obox.center - self.pcd.get_center(),
        #         -self.vec2snr,
        #     )
        #     + ref_plane_axis_position
        # )
        # self.pcd.translate(
        #     translation=np.array([0, sensor_axis_shift, 0]), relative=True
        # )
        # if self.disp_info:
        #     print(
        #         f"[INFO: {ut.currentFuncName()}]: "
        #         f"The pcd's ref_plane now intersect with y-axis at "
        #         f"{sensor_axis_shift}"
        #     )

    def logend_scaling(
        self,
        param_patch_det: ut.ParamPlanarPatchDetection,
        param_logend_det: ut.ParamLogendDetection,
    ) -> None:
        """
        Scale one log-end section so to get log-end diameters
        and distances to the ref_plane

        Args:
            param_patch_det (Type[ut.ParamPlanarPatchDetection])
                -- param for planar patch detection
            param_fid_det (Type[ut.ParamFiducialDetection])
                -- param for fiducial detection
            param_logend_det (Type[ut.ParamLogendDetection])
                -- param for log-end measurement process
            ang_offset(type: float)=np.pi/6 (default)
                -- angle offset tolerance, radian unit
            max_num_planes(type: int)=50 (default)
                -- max number of planes for plane_ransac segmentation

        Returns:
            None

        """
        # Sanity check for the input arguments
        ut.sck.is_valid_pcd(self.pcd, "self.pcd")

        # pick the points whose normal vecs align with self.vec2snr
        aa = np.array(self.pcd.normals)
        bb = np.tile(self.vec2snr, (len(aa), 1))
        dd = ut.vector_angle(aa, bb)
        nv2snr_idx = np.where(
            (dd <= param_logend_det.pose_ang_lo)
            | (dd >= param_logend_det.pose_ang_hi)
        )[0].tolist()
        if nv2snr_idx is not None and len(nv2snr_idx) > 0:
            self.__pcd_nv2snr = self.pcd.select_by_index(nv2snr_idx)
        else:
            self.__pcd_nv2snr = self.pcd

        self.planar_patches(
            the_pcd=self.__pcd_nv2snr,
            # the_pcd=self.pcd,
            normal_variance_threshold_deg=param_patch_det.normal_variance_threshold_deg,  # noqa
            coplanarity_deg=param_patch_det.coplanarity_deg,
            outlier_ratio=param_patch_det.outlier_ratio,
            min_plane_edge_length=param_patch_det.min_plane_edge_len,
            min_num_points=param_patch_det.min_num_pts,
            search_knn=param_patch_det.search_knn,
        )
        if self.disp_info:
            ut.draw_planar_patches(
                oboxes=self.planar_patch_oboxes,
                o3dobj=self.pcd,
                uvw_scale=0.2,
                uvw_selected=np.array([True, True, True]),
                disp_info=self.disp_info,
            )

        self.filter_planar_patches(
            obox_candidates=self.planar_patch_oboxes,
            pose_ang_lo=param_logend_det.pose_ang_lo,
            pose_ang_hi=param_logend_det.pose_ang_hi,
            diag_uv_lo=param_logend_det.diag_uv_lo,
            diag_uv_hi=param_logend_det.diag_uv_hi,
            ratio_uv_lo=param_logend_det.ratio_uv_lo,
            ratio_uv_hi=param_logend_det.ratio_uv_hi,
            grd_hgt_lo=param_logend_det.grd_hgt_lo,
        )

        self.get_log_scaling_results()
