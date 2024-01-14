# ----------------------------------------------------------------------------
# - PointCound data processing utility functions                            -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
from __future__ import annotations
import copy
import os as os
import sys
import numpy as np
from typing import Annotated, Literal, Tuple, TypeAlias, TypeVar
import numpy.typing as npt
from varname import nameof
import open3d as o3d
import open3d.visualization.gui as gui  # noqa
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# typing for the frequently used data types
DT_FLOAT = TypeVar("DT_FLOAT", bound=np.float64)
FloatArray3 = Annotated[npt.NDArray[DT_FLOAT], Literal[3]]
FloatArray4 = Annotated[npt.NDArray[DT_FLOAT], Literal[4]]
FloatArrayN = Annotated[npt.NDArray[DT_FLOAT], Literal["N"]]
FloatArray3x3 = Annotated[npt.NDArray[DT_FLOAT], Literal[3, 3]]
FloatArrayNx3 = Annotated[npt.NDArray[DT_FLOAT], Literal["N", 3]]
FloatArrayNx4 = Annotated[npt.NDArray[DT_FLOAT], Literal["N", 4]]
IntArrayN = Annotated[npt.NDArray[np.int16], Literal["N"]]
BoolArray3 = Annotated[npt.NDArray[np.bool_], Literal[3]]
BoolArrayN = Annotated[npt.NDArray[np.bool_], Literal["N"]]
PointCloud: TypeAlias = o3d.cpu.pybind.geometry.PointCloud
TriangleMesh: TypeAlias = o3d.cpu.pybind.geometry.TriangleMesh
OrientedBoundingBox: TypeAlias = o3d.cpu.pybind.geometry.OrientedBoundingBox


def currentFuncName(n: int = 0) -> str:
    """
    Get the current function name with the executing function

    Args:
        n(type: int) = 0, 1, or 2 (default = 0)
        for current func name, specify 0 or no argument;
        for name of caller of current func, specify 1;
        for name of caller of caller of current func, specify 2. etc.
        for example:
            print(f"You are in function: {currentFuncName()}")
            print(f"This function's caller was: {currentFuncName(1)}")
            print(f"This function's caller's caller was: {currentFuncName(2)}")

    Returns:
        the function name string according to the arg n

    """
    # sanity check
    assert n in np.array([0, 1, 2]), (
        f"[ERROR: currentFuncName]: "
        f"Invalid arg (n={n})! n should be any of [0, 1, 2]. "
    )

    return sys._getframe(n + 1).f_code.co_name


# constant parameters
class ColorPalettes:
    """
    Define common RGB color palettes for the ease of use --
        BLACK, RED, RED_DARK, ORANGE, ORANGE_DARk,
        YELLOW, YELLOW_DARK, GREEN, GREEN_DARK,
        CYAN, CYAN_DARK, BLUE, BLUE_DARK,
        PINK, PINK_DARK, WHITE, GRAY, GRAY_DARK

    """

    BLACK: FloatArray3 = np.array([0, 0, 0]) / 255
    RED: FloatArray3 = np.array([255, 0, 0]) / 255
    RED_DARK: FloatArray3 = np.array([128, 0, 0]) / 255
    ORANGE: FloatArray3 = np.array([255, 165, 0]) / 255
    ORANGE_DARK: FloatArray3 = np.array([128, 83, 0]) / 255
    YELLOW: FloatArray3 = np.array([255, 255, 0]) / 255
    YELLOW_DARK: FloatArray3 = np.array([128, 128, 0]) / 255
    GREEN: FloatArray3 = np.array([0, 255, 0]) / 255
    GREEN_DARK: FloatArray3 = np.array([0, 128, 0]) / 255
    CYAN: FloatArray3 = np.array([0, 255, 255]) / 255
    CYAN_DARK: FloatArray3 = np.array([0, 128, 128]) / 255
    BLUE: FloatArray3 = np.array([0, 0, 255]) / 255
    BLUE_DARK: FloatArray3 = np.array([0, 0, 128]) / 255
    PINK: FloatArray3 = np.array([255, 192, 203]) / 255
    PINK_DARK: FloatArray3 = np.array([128, 96, 102]) / 255
    WHITE: FloatArray3 = np.array([255, 255, 255]) / 255
    GRAY: FloatArray3 = np.array([128, 128, 128]) / 255
    GRAY_DARK: FloatArray3 = np.array([64, 64, 64]) / 255


class ParamPlanarPatchDetection:
    """
    Define parameters for the planar patch detection
    Reference -- http://www.open3d.org/docs/release/jupyter/geometry/pointcloud.html?highlight=detect_planar_patches#Planar-patch-detection     # noqa
    """

    def __init__(
        self,
        normal_variance_threshold_deg: float = 60,  # 60
        coplanarity_deg: float = 75,  # 75
        outlier_ratio: float = 0.75,  # 0.75
        min_plane_edge_len: int = 0,  # 0
        min_num_pts: int = 0,  # 0
        search_knn: int = 30,  # 30
    ):
        self.normal_variance_threshold_deg: float = (
            normal_variance_threshold_deg
        )
        self.coplanarity_deg: float = coplanarity_deg
        self.outlier_ratio: float = outlier_ratio
        self.min_plane_edge_len: float = min_plane_edge_len
        self.min_num_pts: int = min_num_pts
        self.search_knn: int = search_knn


class ParamFiducialDetection:
    """
    Define parameters for detecting fiducials to get a reference plane
    """

    def __init__(
        self,
        fid_patch_ang_lo: float = np.pi / 3,
        fid_patch_ang_hi: float = np.pi - np.pi / 3,
        fid_patch_ratio_uv_lo: float = 0.5,
        fid_patch_ratio_uv_hi: float = 2.0,
    ):
        # setup ref_plane/fiducials detection parameters
        self.fid_patch_ang_lo: float = fid_patch_ang_lo
        self.fid_patch_ang_hi: float = fid_patch_ang_hi
        self.fid_patch_ratio_uv_lo: float = fid_patch_ratio_uv_lo
        self.fid_patch_ratio_uv_hi: float = fid_patch_ratio_uv_hi


class ParamLogendDetection:
    """
    Define log patch detection parameters
    """

    def __init__(
        self,
        pose_ang_lo: float = np.pi / 6,
        pose_ang_hi: float = np.pi - np.pi / 6,
        diag_uv_lo: float = 0.12,
        diag_uv_hi: float = 0.72,
        ratio_uv_lo: float = 0.75,
        ratio_uv_hi: float = 1.25,
        grd_hgt_lo: float = 0.24,
    ):
        self.pose_ang_lo: float = pose_ang_lo
        self.pose_ang_hi: float = pose_ang_hi
        self.diag_uv_lo: float = diag_uv_lo
        self.diag_uv_hi: float = diag_uv_hi
        self.ratio_uv_lo: float = ratio_uv_lo
        self.ratio_uv_hi: float = ratio_uv_hi
        self.grd_hgt_lo: float = grd_hgt_lo


class SanityCheck:
    """
    Sanity check on the data variables
    """

    def __init__(self) -> None:
        pass

    def is_nonzero_vector3(self, vec: FloatArray3, vec_name: str) -> None:
        """
        The input vec should be a 3-ele nonzero vector.
        """
        assert (vec.shape == (3,)) & (np.any(vec != 0.0)), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{vec_name} should be a non-zero 1-3 np.ndarray!"
        )

    def is_different_vector3(
        self, u: FloatArray3, v: FloatArray3, u_name: str, v_name: str
    ) -> None:
        """
        The input u & v 3-ele vectors should be different.
        """
        assert (u.shape == v.shape) & (np.any(v != u)), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{u_name} and {v_name} vectors are exactly the same!"
        )

    def is_positive_int(self, val: int, val_name: str) -> None:
        """
        The input argument should be a positive int.
        """
        assert val > 0, (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {val_name}={val:d}, which should be a positive int!"
        )

    def is_positive_float(self, val: float, val_name: str) -> None:
        """
        The input argument should be a positive float.
        """
        assert val > 0.0, (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {val_name}={val:.2e}, which should be a positive float!"
        )

    def is_float_hi_gt_lo(
        self, lo: float, hi: float, lo_name: str, hi_name: str
    ) -> None:
        """
        The input argument hi should be greater than lo.
        """
        assert hi > lo, (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{hi_name} = {hi:.2e} has to be greater than {lo_name} = {lo:.2e})"
        )

    def is_float_value_bounded(
        self, val: float, lo: float, hi: float, val_name: str
    ) -> None:
        """
        The input argument val(float) should be witin [lo, hi].
        """
        assert hi > lo, (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"lo = {lo:.2e} should be less than hi = {hi:.2e}"
        )
        assert (val >= lo) & (val <= hi), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{val_name} = {val:.2e}, which should be within "
            f"[{lo:.2e}, {hi:.2e}]"
        )

    def is_int_value_bounded(
        self, val: int, lo: int, hi: int, val_name: str
    ) -> None:
        """
        The input argument val(int) should be within [lo, hi].
        """
        assert hi > lo, (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"lo = {lo} should be less than hi = {hi}"
        )
        assert (val >= lo) & (val <= hi), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{val_name} = {val}, which should be within [{lo}, {hi}]"
        )

    def is_valid_positive_float_bound(
        self, lo: float, hi: float, lo_name: str, hi_name: str
    ) -> None:
        """
        The input argument should be a [lo, hi] bound of float values.
        """
        assert (lo >= 0.0) & (lo < hi), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{lo_name} = {lo:.2e}, {hi_name} = {hi:.2e}"
            f" are non-conformant!"
        )

    def is_valid_positive_int_bound(
        self, lo: int, hi: int, lo_name: str, hi_name: str
    ) -> None:
        """
        The input argument should be a [lo, hi] bound of int values.
        """
        assert (lo >= 0) & (lo < hi), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"{lo_name} = {lo}, {hi_name} = {hi} are non-conformant!"
        )

    def is_valid_pcd(self, pcd: PointCloud, pcd_name: str) -> None:
        """
        The input argument should be a valid pcd that contains 3D points.
        """
        assert pcd.has_points(), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {pcd_name} is has no 3D points!"
        )

    def is_valid_mesh(self, mesh: TriangleMesh, mesh_name: str) -> None:
        """
        The input argument should be a valid mesh that contains verticies.
        """
        assert mesh.has_vertices(), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {mesh_name} is has no vertices!"
        )

    def is_valid_obox(self, obox: OrientedBoundingBox, obox_name: str) -> None:
        """
        The input argument should be a valid pcd that contains 3D points.
        """
        assert (
            obox.is_empty() is False
        ), f"[ERROR: {currentFuncName(n=1)}]: {obox_name} is empty!"

    def is_valid_o3dobj_list(
        self,
        o3dobjs: list[PointCloud | TriangleMesh | OrientedBoundingBox],
        o3dobjs_name: str,
        selected_idx: list[bool],
    ) -> None:
        """
        The input argument should be a list of o3dobjs, with the selected_idx
            specifying the pickup.
        """
        assert (
            len(o3dobjs) > 0
        ), f"[ERROR: {currentFuncName(n=1)}]: {o3dobjs_name} is not a list!"
        assert len(selected_idx) == len(o3dobjs), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"selected_idx has different lenth as {o3dobjs_name}!"
        )
        o3dobjs_selected = (np.asarray(o3dobjs)[selected_idx]).tolist()
        # Sanity check for each selected obox
        assert (
            len(o3dobjs_selected) > 0
        ), f"[ERROR: {currentFuncName(n=1)}]: o3dobjs_selected is not a list!"
        for jj, obox in enumerate(o3dobjs_selected):
            self.is_valid_obox(obox, f"{o3dobjs_name}[{selected_idx[jj]}]")

    def is_valid_plane_model(
        self, plane_model: FloatArray4, plane_model_name: str
    ) -> None:
        """
        The input argument should be a valid 4-ele plane model equation
            coefficient vector -- [a, b, c, d] in ax + by + cz + d = 0
        """
        assert np.any(plane_model[0:3] != 0.0), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {plane_model_name} is non-conformant!"
        )

    def is_valid_color_vector3(
        self, color_vec: FloatArray3, color_vec_name: str
    ) -> None:
        """
        The input argument should be a 3-ele color [r, g, b] vector with
            all elements within [0.0, 1.0] range.
        """
        assert np.all(color_vec >= 0.0) & np.all(color_vec <= 1.0), (
            f"[ERROR: {currentFuncName(n=1)}]: "
            f"Input arg {color_vec_name} need to be of 1-3 shape with all "
            f"[r, g, b] elements within [0.0, 1.0]!"
        )

    def is_valid_file_path(self, file_path: str) -> None:
        """
        The input argument should be a string that contains a valid file path
        """
        assert os.path.isfile(file_path), (
            f"[INFO: {currentFuncName()}]: "
            f"Undefined or non-existing file {file_path}!"
        )


CP: ColorPalettes = ColorPalettes()
sck: SanityCheck = SanityCheck()
inch2mm: float = 25.4
foot2inch: float = 12.0
m2mm: float = 1000.0
xaxis: np.ndarray = np.array([1.0, 0.0, 0.0])
yaxis: np.ndarray = np.array([0.0, 1.0, 0.0])
zaxis: np.ndarray = np.array([0.0, 0.0, 1.0])
xyplane: np.ndarray = np.array([0.0, 0.0, 1.0, 0.0])
yzplane: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
zxplane: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0])


def get_xyz_axes(
    frame_size: float = 1.0,
    frame_origin: FloatArray3 = np.array([0.0, 0.0, 0.0]),
) -> TriangleMesh:
    """
    Get the xyz axes mesh_frame in the coordinates with the
    frame_size (default=1.0) and frame_origin (defalut=[0.0, 0.0, 0.0])

    Args:
        frame_size = 1.0 (default)
            -- mesh frame size, must be a positive float
        frame_origin = np.array([0.0, 0.0, 0.0]) (default)
            -- mesh origin [x, y, z]

    Returns:
        mesh(type: o3d.TriangleMesh) -- the o3d triangle mesh for the xyz axes

    """
    # sanity check
    sck.is_positive_float(frame_size, nameof(frame_size))

    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=frame_origin
    )


def get_arrow_mesh(
    origin: FloatArray3 = np.array([0, 0, 0]),
    vector: FloatArray3 = np.array([0, 0, 1]),
    scale: float = 1.0,
    arrow_color: FloatArray3 = CP.RED,
    cylinder_radius: float = 0.02,
    cylinder_height: float = 1.0,
    cylinder_split: int = 4,
    cone_radius: float = 0.04,
    cone_height: float = 0.24,
    cone_split: int = 1,
    resolution: int = 20,
) -> TriangleMesh:
    """
    Creates an arrow mesh from an origin point to an end point,
    or create an arrow pointing to the vec from origin.

    Args:
        origin: arrow origin (type: np.array([x, y, z]))
        vector: arrow end-point (type: np.array([x, y, z]))
        scale: arrow mesh size scaling factor (positive float)
        arrow_color: arrow mesh solor ((type: np.array([r, g, b]),
            default=np.array([1.0, 0.0, 0.0]))
        cylinder_radius: arrow cylinder diameter (positive float, default=0.02)
        cylinder_height: arrow cylinder height (positive float, default=1.0)
        cylinder_split: arrow cylinder mesh split (positive int, default=4)
        cone_radius: arrow head diameter (positive float, default=0.04)
        cone_height: arrow head height (positive float, default=0.24)
        cone_split: arrow head mesh split (positive int, default=1)
        resolution: arrow circumference split (positive int, default=20)

    Returns:
        mesh_arrow: an Open3D TriangleMesh object

    """
    # Sanity check
    sck.is_different_vector3(origin, vector, nameof(origin), nameof(vector))
    sck.is_valid_color_vector3(arrow_color, nameof(arrow_color))
    sck.is_positive_float(scale, nameof(scale))
    sck.is_positive_float(cylinder_radius, nameof(cylinder_radius))
    sck.is_positive_float(cylinder_height, nameof(cylinder_height))
    sck.is_positive_int(cylinder_split, nameof(cylinder_split))
    sck.is_positive_float(cone_radius, nameof(cone_radius))
    sck.is_positive_float(cone_height, nameof(cone_height))
    sck.is_positive_int(cone_split, nameof(cone_split))
    sck.is_positive_int(resolution, nameof(resolution))

    Ry = Rz = np.eye(3)
    # T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # T[:3, -1] = origin

    nvec = np.array(vector - origin)
    nvec = nvec / vector_magnitude(nvec)
    Rz, Ry = calculate_zy_rotation_for_arrow(nvec)

    # Create the arrow mesh
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=cone_radius * scale,
        cone_height=cone_height * scale,
        cylinder_radius=cylinder_radius * scale,
        cylinder_height=cylinder_height * scale,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    mesh_arrow.paint_uniform_color(arrow_color)
    mesh_arrow.rotate(R=Ry, center=np.array([0, 0, 0]))
    mesh_arrow.rotate(R=Rz, center=np.array([0, 0, 0]))
    mesh_arrow.translate(translation=origin, relative=True)

    return mesh_arrow


def vector_magnitude(vec: FloatArrayNx3) -> FloatArrayN | float:
    """
    Calculate the element-wise magnitude for a scalar or a N-D array

    Args:
        vec: a N-D array e.g. np.ndarray([[x1, y1, z1], ..., [xn, yn, zn]])

    Returns:
        Element-wise vec magnitude along its horizontal axis

    """
    if vec.ndim == 1:
        mag: float = (float)(np.linalg.norm(vec))
        return mag
    else:
        return np.array(
            np.linalg.norm(vec, axis=vec.ndim - 1), dtype=np.float64
        )


def vector_angle(u: FloatArrayNx3, v: FloatArrayNx3) -> FloatArrayN | float:
    """
    Calculate the element-wise angle between two N-D vector arrays

    Args:
        u, v: two N-3 vector arrays of the same shape
        (type: np.array([x1, y1, z1], ..., [xn, yn, zn]))

    Returns:
        The element-wise angle between u and v, of radian unit
        (type: np.array([ang1, ang2, ..., angn]))

    Reference: https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy   # noqa

    """
    # Sanity check
    assert u.shape == v.shape, (
        f"[INFO: {currentFuncName()}]: "
        f"Invalid args u, v! u, v should be of the same shape."
    )

    if u.ndim == 1:
        sck.is_nonzero_vector3(u, nameof(u))
        sck.is_nonzero_vector3(v, nameof(v))
        ww = np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v))
        ww = np.round(ww, 6)
        assert np.abs(ww) <= 1.0, (
            f"[INFO: {currentFuncName()}]: "
            f"The calculated dot(u, v) should be within [-1.0, 1.0]!"
        )
        return (float)(np.arccos(ww))
    else:
        assert (np.all(np.linalg.norm(u, axis=1) > 0.0)) & (
            np.all(np.linalg.norm(v, axis=1) > 0.0)
        ), (
            f"[ERROR: {currentFuncName()}]: "
            f"Invalid args u, v! u, v should all be non-zero vectors."
        )
        nu = u / np.repeat(
            np.linalg.norm(u, axis=1)[:, np.newaxis], u.shape[1], 1
        )
        nv = v / np.repeat(
            np.linalg.norm(v, axis=1)[:, np.newaxis], v.shape[1], 1
        )
        ww = np.einsum("ij,ij->i", nu, nv)
        # ww = np.sum(nu * nv, axis=1)  # another way to do it
        # ww = [np.dot(r1, r2) for (r1, r2) in zip(nu, nv)]  # dot loop way

        # prevent from out-of-range errors for the arccos calc
        ww = np.round(ww, 6)
        assert np.all(np.abs(ww) <= 1.0), (
            f"[INFO: {currentFuncName()}]: "
            f"All calculated dot(u, v) elements should be within [-1.0, 1.0]!"
        )
        return np.array(np.arccos(ww), dtype=np.float64)


def vec_proj_on_plane(
    u: FloatArrayNx3, plane_model: FloatArrayNx4
) -> FloatArrayNx3 | FloatArray3:
    """
    Calculates the element-wise projection vector array onto an array of planes
        defined by their 4-element plane models.

    Args:
        u: N-3 vector np.array
        plane_model: N-4 vector np.array

    Returns:
        norm_vec: The element-wise plane normal unit vector array between u, v
        (type: np.array([nx1, ny1, nz1], ..., [nxn, nyn, nzn]))

    """
    # Sanity check
    assert ((u.ndim == 1) & (plane_model.ndim == 1)) | (
        (u.ndim == 2) & (plane_model.ndim == 2) & (len(u) == len(plane_model))
    ), (
        f"[INFO: {currentFuncName()}]: "
        f"Invalid args u, plane_model, which should have the same length."
    )

    if plane_model.ndim == 1:
        sck.is_nonzero_vector3(u, nameof(u))
        sck.is_valid_plane_model(plane_model, nameof(plane_model))
        pnvec = np.array(plane_model)[0:3]
    else:
        for jj, (vec, p_model) in enumerate(zip(u, plane_model)):
            sck.is_nonzero_vector3(vec, f"u[{jj}]")
            sck.is_valid_plane_model(p_model, f"plane_model[{jj}]")
        pnvec = np.array(plane_model)[:, 0:3]

    u_on_pnvec = (np.dot(u, pnvec) / np.dot(pnvec, pnvec)) * pnvec

    return np.array(u - u_on_pnvec)


def vec_norm2plane(u: FloatArrayNx3, v: FloatArrayNx3) -> FloatArrayNx3:
    """
    Calculates the element-wise plane normal vector array of planes defined
    by two vector arrays.

    Args:
        u, v: two N-3 arrays of the same shape
        (type: np.array([x1, y1, z1], ..., [xn, yn, zn]))

    Returns:
        norm_vec: The element-wise plane normal unit vector array between u, v
        (type: np.array([nx1, ny1, nz1], ..., [nxn, nyn, nzn]))

    """
    # Sanity check
    assert u.shape == v.shape, (
        f"[INFO: {currentFuncName()}]: "
        f"Invalid args u, v! u, v should be of the same shape."
    )

    if u.ndim == 1:
        sck.is_nonzero_vector3(u, nameof(u))
        sck.is_nonzero_vector3(v, nameof(v))
        w = np.cross(u, v)
        assert vector_magnitude(w) > 0.0, (
            f"[INFO: {currentFuncName()}]: "
            f"Norm vector magnitude w={w}! w should be a positive float."
        )
        return np.array(w / vector_magnitude(w), dtype=np.float64)
    else:
        w = np.cross(u, v, axisa=1, axisb=1)
        assert np.all(vector_magnitude(w) > 0.0), (
            f"[INFO: {currentFuncName()}]: "
            f"Norm vector magnitude array elements should all be positive!"
        )
        return np.array(
            w / (np.array(vector_magnitude(w)).T).reshape(-1, 1),
            dtype=np.float64,
        )


def rotmat_uu2vv(uu: FloatArray3, vv: FloatArray3) -> FloatArray3x3:
    """
    Find the rotation matrix that aligns uu to vv

    Args:
        uu: A "source" vector (1x3 size)
        vv: A "destination" vector (1x3 size)

    Returns:
        rot_mat: a transform matrix (3x3 size) which when
            applied to uu, aligns uu with vv.

    """
    # Sanity check
    assert (uu.shape == vv.shape) & (uu.shape == (3,)), (
        f"[INFO: {currentFuncName()}]: "
        f"Invalid args uu, vv! uu, vv should be of the same 1-3 size."
    )
    assert np.any(uu != 0.0) & np.any(uu != 0.0), (
        f"[INFO: {currentFuncName()}]: "
        f"Invalid args uu, vv! uu or vv should not be a zero vector!"
    )

    a, b = (uu / np.linalg.norm(uu)).reshape(3), (
        vv / np.linalg.norm(vv)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    return np.array(rot_mat)


def calculate_zy_rotation_for_arrow(
    vec: FloatArray3,
) -> Tuple[FloatArray3x3, FloatArray3x3]:
    """
    Calculate the rotations required to go from the vector vec to the z-axis.
    The first rotation that is calculated is over the z axis. This will leave
    the vec on the XZ plane. Then, the rotation over the y-axis.

    Args:
        vec: a np.array([u, v, w]) for the current vec

    Returns:
        Rz, Ry: two rotation matrices of [3, 3] size

    """
    # Sanity check
    sck.is_nonzero_vector3(vec, nameof(vec))

    # Rotation over z axis of the FOR
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )

    return (Rz, Ry)


def calc_cuboid_vert8(
    center: FloatArray3,
    uu: FloatArray3,
    vv: FloatArray3,
    ww: FloatArray3,
    lu: float,
    lv: float,
    lw: float,
) -> FloatArrayNx3:
    """
    Find the rotation matrix that aligns uu to vv

    Args:
        center: the cuboid center (x, y, z)
        uu, vv, ww: three edge unit vectors
        lu, lv, lw: three edge lengths

    Returns:
        vert8: the 8 vertices of the cuboid

    """
    # Sanity check
    sck.is_nonzero_vector3(uu, nameof(uu))
    sck.is_nonzero_vector3(vv, nameof(vv))
    sck.is_nonzero_vector3(ww, nameof(ww))
    sck.is_positive_float(lu, nameof(lu))
    sck.is_positive_float(lv, nameof(lv))
    sck.is_positive_float(lw, nameof(lw))

    c1, c2 = center - ww * lw, center + ww * lw
    pu, pv = lu * uu, lv * vv
    vert8 = np.array(
        [
            c1 + (-pv - pu),
            c1 + (pv - pu),
            c1 + (pv + pu),
            c1 + (-pv + pu),
            c2 + (-pv + pu),
            c2 + (pv + pu),
            c2 + (pv - pu),
            c2 + (-pv - pu),
        ]
    )

    return vert8


def estimate_dbscan_epsilon(
    pcd: PointCloud,
    nth_neighbors: int = 20,
    dbscan_epsilon_default: float = 0.001,
    disp_info: bool = True,
) -> float:
    """
    Estimated the distance curve among nearest neighbor points, and select the
    appropriate dbscan_epsilon value for doing dbscan cluster segmentation
    on a given point cloud.

    Args:
        pcd(type: o3d.PointCloud)
        nth_neighbors(type: int)=20 (default)
        default_dbscan_epsilon(type: float)=0.001 (default)

    Returns:
        dbscan_epsilon(type: float)

    Reference-- https://github.com/rehmantalha1999/dbscan_ransan/blob/main/version%203.ipynb  # noqa

    """
    # Sanity check for the input arguments
    sck.is_valid_pcd(pcd, "pcd")
    sck.is_positive_int(nth_neighbors, nameof(nth_neighbors))
    sck.is_positive_float(dbscan_epsilon_default, "dbscan_epsilon_default")

    # Further getting the pcd.points as np array
    dataset = np.array(pcd.points)

    # Compute distances to n_neighbors_th nearest neighbor
    neighbors = NearestNeighbors(n_neighbors=nth_neighbors)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Compute curvature
    dx = 1
    dy = np.gradient(distances, dx)
    d2y = np.gradient(dy, dx)
    curvature = np.abs(d2y) / (1 + dy**2) ** (3 / 2)

    # Find point of maximum curvature
    max_curvature_idx = np.argmax(curvature)
    max_curvature_point = (max_curvature_idx, distances[max_curvature_idx])

    # Print epsilon value and plot curve & maximum curvature point
    if disp_info:
        print(
            f"[INFO: {currentFuncName()}]: "
            f"estimated dbscan_epsilon = {max_curvature_point[1]:.2e}"
        )
        plt.plot(distances)
        plt.plot(*max_curvature_point, "ro")
        plt.show()

    if max_curvature_point[1] <= 0:
        dbscan_epsilon = dbscan_epsilon_default
        if disp_info:
            print(
                f"[INFO: {currentFuncName()}]: "
                f"dbscan_epsilon{max_curvature_point[1]:.4f} estimated as "
                f"a non-positive float, setting it as "
                f"({dbscan_epsilon_default:.2e})"
            )
    else:
        dbscan_epsilon = max_curvature_point[1]

    return dbscan_epsilon


def get_best_distance_threshold(
    pcd: PointCloud,
    distance_threshold_default: float = 0.01,
    disp_info: bool = True,
) -> float:
    """
    Calculates the best distance threshold and set distance_threshold
    value for a given point cloud.

    Args:
        pcd(type: o3d.PointCloud)=None (default)
        distance_threshold_default(type: float)=0.01 (default)

    Returns:
        distance_threshold(type: float)

    Reference-- https://github.com/rehmantalha1999/dbscan_ransan/blob/main/version%203.ipyn  # noqa

    """
    # Sanity check for the input arguments
    sck.is_valid_pcd(pcd, "pcd")
    sck.is_positive_float(
        distance_threshold_default, "distance_threshold_default"
    )

    distances = pcd.compute_nearest_neighbor_distance()
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + 0.5 * std_dist
    if disp_info:
        print(
            f"[INFO: {currentFuncName()}]: "
            f"distance_threshold = {threshold:.2e} with mean_dist"
            f" / std_ dist = {mean_dist:.2e} / {std_dist:.2e}"
        )
    if threshold <= 0:
        distance_threshold = distance_threshold_default
        if disp_info:
            print(
                f"[INFO: {currentFuncName()}]: "
                f"distance_threshold{distance_threshold:.2e} estimated as "
                f"a non-positive float, setting it as "
                f"({distance_threshold_default:.2e})"
            )
    else:
        distance_threshold = threshold

    return distance_threshold


def load_pcd_from_mesh(
    mesh_path: str, disp_info: bool = True
) -> Tuple[PointCloud, TriangleMesh]:
    """
    Load mesh from a file and extract its point cloud.

    Args:
        mesh_path(type: str) -- mesh file path
        disp_info(type: bool) -- display info or not (default=True)

    Returns:
        pcd(type: o3d.PointCloud)
        mesh(type: o3d.TriangleMesh)

    """
    # Sanity check
    sck.is_valid_file_path(file_path=mesh_path)

    if disp_info:
        print(f"[INFO: {currentFuncName()}]: Loading mesh {mesh_path} ...")
    mesh = o3d.io.read_triangle_mesh(
        filename=mesh_path,
        enable_post_processing=True,
        print_progress=disp_info,
    )
    # extract pcd from mesh
    sck.is_valid_mesh(mesh, "mesh")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    ctr, bmn, bmx = get_o3dobj_info(o3dobj=mesh, disp_info=disp_info)
    ctr, bmn, bmx = get_o3dobj_info(o3dobj=pcd, disp_info=disp_info)

    return (pcd, mesh)


def load_pcd(pcd_path: str, disp_info: bool = True) -> PointCloud:
    """
    Load point cloud from a file.

    Args:
        pcd_path(type: string) -- pcd file path
        disp_info(type: bool) -- display info or not (default=True)

    Returns:
        pcd(type: o3d.PointCloud)

    """
    # Sanity check
    sck.is_valid_file_path(file_path=pcd_path)

    if disp_info:
        print(f"[INFO: {currentFuncName()}]: Loading pcd {pcd_path} ...")
    # load mesh into Open3D
    pcd = o3d.io.read_point_cloud(
        filename=pcd_path,
        format="auto",
        remove_nan_points=True,
        remove_infinite_points=True,
        print_progress=disp_info,
    )
    sck.is_valid_pcd(pcd, "pcd")
    if pcd.has_normals() is False:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.01, max_nn=30
            )
        )

    ctr, bmn, bmx = get_o3dobj_info(o3dobj=pcd, disp_info=disp_info)

    return pcd


def get_o3dobj_info(
    o3dobj: PointCloud | TriangleMesh,
    o3dobj_name: str = "",
    disp_info: bool = True,
) -> Tuple[FloatArray3, FloatArray3, FloatArray3]:
    """
    Get basic info about a given mesh or point cloud.

    Args:
        o3dobj(type: o3d.PointCloud | o3d.TriangleMesh)=None (default)

    Returns:
        ctr(type: np.array([xc, yc, zc]) -- o3dobj center position
        bmn(type: np.array([xmn, ymn, zmn]) -- o3dobj min bound
        bmx(type: np.array([xmx, ymx, zmx]) -- o3dobj max bound

    """
    # Sanity check for the input arguments
    assert (
        o3dobj is not None
    ), f"[ERROR: {currentFuncName()}]: o3dobj undefined!"

    ctr = o3dobj.get_center()
    bmn = o3dobj.get_min_bound()
    bmx = o3dobj.get_max_bound()
    if disp_info:
        if isinstance(o3dobj, o3d.cpu.pybind.geometry.PointCloud):
            o3dobj_name = "pcd" if o3dobj_name == "" else o3dobj_name
            print(
                f"[INFO: {currentFuncName()}]: "
                f"{o3dobj_name} info -- \n"
                f"     pts {np.asarray(o3dobj.points).shape[0]:,}\n"
                f"     ctr ({ctr[0]:.2e}, {ctr[1]:.2e}, {ctr[2]:.2e})\n"
                f"     min_bound ({bmn[0]:.2e}, {bmn[1]:.2e}, {bmn[2]:.2e})\n"
                f"     max_bound ({bmx[0]:.2e}, {bmx[1]:.2e}, {bmx[2]:.2e})\n"
                f"     has normals: {o3dobj.has_normals()} "
            )

        if isinstance(o3dobj, o3d.cpu.pybind.geometry.TriangleMesh):
            o3dobj_name = "mesh" if o3dobj_name == "" else o3dobj_name
            print(
                f"[INFO: {currentFuncName()}]: "
                f"{o3dobj_name} info -- \nvert {len(o3dobj.vertices):,}, "
                f"tri {len(o3dobj.triangles):,}\n"
                f"     ctr ({ctr[0]:.2e}, {ctr[1]:.2e}, {ctr[2]:.2e})\n"
                f"     min_bound ({bmn[0]:.2e}, {bmn[1]:.2e}, {bmn[2]:.2e})\n"
                f"     max_bound ({bmx[0]:.2e}, {bmx[1]:.2e}, {bmx[2]:.2e})\n"
            )

    return (ctr, bmn, bmx)


def get_o3dobj_obox_and_uvw(
    o3dobj: PointCloud | TriangleMesh | OrientedBoundingBox,
    obox_color: FloatArray3 = CP.ORANGE,
    uvw_scale: float = 1.0,
    uvw_selected: BoolArray3 = np.array([True, True, True]),
    mu_color: FloatArray3 = CP.RED_DARK,
    mv_color: FloatArray3 = CP.GREEN_DARK,
    mw_color: FloatArray3 = CP.BLUE_DARK,
) -> Tuple[OrientedBoundingBox, TriangleMesh]:
    """
    Get the oriented bounding box of a point cloud or triangle mesh with
    a given display color, as well as its local u, v, w vector meshes

    Args:
        o3dobj(type: o3d.PointCloud | o3d.TriangleMesh |
            o3d.OrientedBoundingBox) -- the o3dobj for display
        obox_color(type: np.array([r, g, b]))=CP.ORANGE (defalut)
            -- set o3dobj's display color
        uvw_scale(type: float)=1.0 (default)
            -- set uvw arrow mesh scale
        uvw_selected(type: list[bool])=[True, True, True] (default)
            -- control which uvw arrow mesh to display
        mu_color(type: np.array([r, g, b]))=CP.RED_DARK (defalut)
        mv_color(type: np.array([r, g, b]))=CP.GREEN_DARK (defalut)
        mw_color(type: np.array([r, g, b]))=CP.BLUE_DARK (defalut)
            -- set uvw arrow mesh color

    Returns:
        obox(type: o3d.OrientedBoundingBox): pcd | mesh oriented bounding box
        uvw_mesh(type: o3d.TriangleMesh | None): uvw-axis mesh for display
            or nothing if uvw_selected is all False

    """
    # Sanity check for the input arguments
    sck.is_positive_float(uvw_scale, nameof(uvw_scale))
    sck.is_valid_color_vector3(obox_color, nameof(obox_color))
    sck.is_valid_color_vector3(mu_color, nameof(mu_color))
    sck.is_valid_color_vector3(mv_color, nameof(mv_color))
    sck.is_valid_color_vector3(mw_color, nameof(mw_color))

    obox: OrientedBoundingBox
    if isinstance(o3dobj, o3d.cpu.pybind.geometry.PointCloud):
        sck.is_valid_pcd(o3dobj, "pcd")
        obox = o3dobj.get_oriented_bounding_box(robust=True)
    elif isinstance(o3dobj, o3d.cpu.pybind.geometry.TriangleMesh):
        sck.is_valid_mesh(o3dobj, "mesh")
        obox = o3dobj.get_oriented_bounding_box(robust=True)
    elif isinstance(o3dobj, o3d.cpu.pybind.geometry.OrientedBoundingBox):
        obox = o3dobj
    else:
        obox = get_xyz_axes(frame_size=0.5).get_oriented_bounding_box()
        print(
            f"[INFO: {currentFuncName()}]: "
            f"o3dobj type unsupported, generated an obox as fail-safe results"
        )

    sck.is_valid_obox(obox, "obox")
    obox.color = obox_color
    obox_ctr = obox.get_center()
    uu, vv, ww = obox.R[:, 0], obox.R[:, 1], obox.R[:, 2]
    if np.any(uvw_selected):
        mu = get_arrow_mesh(obox_ctr, obox_ctr + uu, uvw_scale, mu_color)
        mv = get_arrow_mesh(obox_ctr, obox_ctr + vv, uvw_scale, mv_color)
        mw = get_arrow_mesh(obox_ctr, obox_ctr + ww, uvw_scale, mw_color)
        uvw_mesh = np.sum(np.array([mu, mv, mw])[np.array(uvw_selected)])
    else:
        uvw_mesh = TriangleMesh()

    return (obox, uvw_mesh)


def crop_pcd_voi(pcd: PointCloud) -> None:
    """
    Crop the volume of interest for a given point cloud using
        keyboard/mouse interactions.

    Args:
        pcd(type: o3d.PointCloud)

    Returns:
        None

    """
    # Sanity check for the input arguments
    sck.is_valid_pcd(pcd, "pcd")
    print(
        f"[INFO: {currentFuncName()}]: "
        f"Manual geometry cropping to get PCD volume of interest(VOI)\n"
        f"1) Press 'X|Y|Z' twice to align geometry with minus x|y|z-axis\n"
        f"2) Press 'K' to lock screen and to switch to selection mode\n"
        f"3) Drag for rectangle or ctrl+left click for polygon selection\n"
        f"4) Press 'C' to get a selected geometry\n"
        f"5) Press 'S' to save the selected geometry\n"
        f"6) Press 'F' to switch to freeview mode"
    )
    o3d.visualization.draw_geometries_with_editing([pcd])


def assemble_pcd_clusters(
    pcd_clusters: list[PointCloud],
    kdtree_radius: float = 0.05,
    kdtree_max_nn: int = 30,
) -> Tuple[PointCloud, list[int]]:
    """
    Assemble multiple point cloud clusters into a single point cloud,
        and retain the cluster label info.

    Args:
        pcd_clusters(type: list[o3d.PointCloud]
        kdtree_radius(type: float)=0.1 (default)
        kdtree_max_nn(type: int)=30 (default)

    Returns:
        assembly_pcd(type: o3d.PointCloud)
        assembly_labels(type: np.ndarray[int...])

    """
    # Sanity check for the input arguments
    sck.is_positive_float(kdtree_radius, nameof(kdtree_radius))
    sck.is_positive_int(kdtree_max_nn, nameof(kdtree_max_nn))
    sck.is_valid_o3dobj_list(
        o3dobjs=pcd_clusters,
        o3dobjs_name="pcd_clusters",
        selected_idx=[True] * len(pcd_clusters),
    )

    # Assemble the pcd_clusters into one single pcd
    assembly_pcd = PointCloud()
    assembly_labels = []
    label_id = 0
    for jj, this_cluster in enumerate(pcd_clusters):
        if this_cluster.has_points():
            assembly_pcd = assembly_pcd + this_cluster
            assembly_labels += [label_id] * len(this_cluster.points)
            label_id += 1

    assembly_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=kdtree_radius, max_nn=kdtree_max_nn
        )
    )

    return (assembly_pcd, assembly_labels)


def display_labeled_clusters(
    pcd: PointCloud, cluster_labels: list[int] = []
) -> None:
    """
    Display point cloud clusters.

    Args:
        pcd(type: o3d.PointCloud) -- point cloud
        cluster_labels(type: np.ndarray[int...]) -- cluster labels

    Returns:
        None

    """
    sck.is_valid_pcd(pcd, nameof(pcd))
    if cluster_labels is None or len(cluster_labels) == 0:
        cluster_labels = [0] * len(pcd.points)
    assert len(pcd.points()) == len(
        cluster_labels
    ), f"[ERROR: {currentFuncName()}]: pcd and cluster labels length mismatch!"

    n_clusters = len(set(cluster_labels))
    colors = np.array(
        plt.get_cmap("tab20")(
            np.array(cluster_labels) / (n_clusters if n_clusters > 0 else 1)
        )
    )
    colors[np.where(np.array(cluster_labels) < 0), :] = np.array(
        [0.0, 0.0, 0.0]
    )
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw([pcd, get_xyz_axes(frame_size=0.5)])


def display_inlier_outlier(
    pcd: PointCloud,
    index: list[int],
    color_outlier: FloatArray3 = CP.RED,
    color_inlier: FloatArray3 = CP.GRAY_DARK,
) -> None:
    """
    Display point cloud inliers and outliers

    Args:
        pcd(type: o3d.PointCloud)
        index(type: list[int])
        color_outlier(type: np.array([r, g, b]))=np.array([1.0, 0.0, 0.0])
        color_inlier(type: np.array([r, g, b]))=np.array([0.8, 0.8, 0.8])

    Returns:
        None

    """
    sck.is_valid_pcd(pcd, nameof(pcd))
    if index is None or len(index) == 0:
        index = (list)(range(len(pcd.points)))
    assert len(index) <= len(pcd.points), (
        f"[ERROR: {currentFuncName()}]: "
        f"index length shoud not be greater than the pcd point length!"
    )

    inlier_cloud = pcd.select_by_index(index)
    inlier_cloud.paint_uniform_color(color_inlier)
    outlier_cloud = pcd.select_by_index(index, invert=True)
    outlier_cloud.paint_uniform_color(color_outlier)
    o3d.visualization.draw(
        [inlier_cloud, outlier_cloud, get_xyz_axes(frame_size=0.5)]
    )


def rigid_transform_3D(
    A: FloatArray3x3, B: FloatArray3x3
) -> Tuple[FloatArray3x3, FloatArray3]:
    """
    Calculate the rigid transform matrix between A, B

    Args:
        A: 3x3 matrix of points (usually 3 column vectors)
        B: 3x3 matrix of points (usually 3 column vectors)

    Returns:
        R = 3x3 rotation matrix
        t = 3x1 column vector

    Reference: https://nghiaho.com/?page_id=671

    """

    assert A.shape == B.shape, (
        f"[ERROR: {currentFuncName()}]: "
        f"Input args A & B need to be of the same shape!"
    )

    assert (A.shape[0] == 3) & (B.shape[0] == 3), (
        f"[ERROR: {currentFuncName()}]: "
        f"Input args A & B both need to be of 3-N shape, it is now A.shape = "
        f"{A.shape} B.shape = {B.shape}"
    )

    # find mean
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # H_rank = np.linalg.matrix_rank(H)
    # if H_rank < 3:
    #     raise ValueError(f"rank of H = {H_rank}, expecting 3")

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print(
            f"[INFO: {currentFuncName()}]: "
            "det(R) < R, reflection detected!, correcting for it ..."
        )
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # t is a 3-1 2D array
    t = -R @ centroid_A + centroid_B
    # change it back to a vector3
    t = [t[0][0], t[1][0], t[2][0]]

    return (R, t)


def get_rotation_matrix_from_vectors(
    vec1: FloatArray3 = zaxis, vec2: FloatArray3 = zaxis
) -> FloatArray3x3:
    """
    Create a rotation matrix that rotates vec1 to vec2

    Args:
        vec1(type: np.ndarray) = np.array([0, 0, 1]) (default)
        vec2(type: np.ndarray) = np.array([0, 0, 1]) (default)

    Returns:
        vec1-to-vec2 rotation matrix

    Reference: https://stackoverflow.com/q/63525482/6010071

    """
    # Sanity check
    sck.is_nonzero_vector3(vec1, nameof(vec1))
    sck.is_nonzero_vector3(vec2, nameof(vec2))
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    sck.is_nonzero_vector3(v, nameof(v))
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotmat = np.array(
        np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2)),
        dtype=np.float64,
    )

    return rotmat


def get_oriented_bounding_boxes_info(
    oboxes: list[OrientedBoundingBox],
) -> Tuple[
    FloatArrayNx3,
    FloatArray3,
    FloatArray3,
    FloatArray3,
    FloatArray3x3,
    FloatArray3x3,
    FloatArray3x3,
]:
    """
    Get the center, box dimensions and pose vectors from a list of
        oriented bounding boxes.

    Args:
        oboxes(type: list[o3d.BoundingBox])

    Returns:
        ctrs(type: np.array([[x, y, z], ..., [x, y, z]])
        ev, eu, ew(type: np.array([float, ..., float])
        uu, vv, ww(type: np.array([[x, y, z], ..., [x, y, z]])

    """
    # Sanity check for the input arguments
    sck.is_valid_o3dobj_list(
        o3dobjs=oboxes,
        o3dobjs_name=nameof(oboxes),
        selected_idx=[True] * len(oboxes),
    )

    ctrs = np.array([obox.get_center() for obox in oboxes])
    eu = np.array([obox.extent[0] for obox in oboxes])
    ev = np.array([obox.extent[1] for obox in oboxes])
    ew = np.array([obox.extent[2] for obox in oboxes])
    uu = np.array([obox.R[:, 0].T for obox in oboxes])
    vv = np.array([obox.R[:, 1].T for obox in oboxes])
    ww = np.array([obox.R[:, 2].T for obox in oboxes])

    return (ctrs, eu, ev, ew, uu, vv, ww)


def filter_oriented_bounding_boxes(
    obox_candidates: list[OrientedBoundingBox],
    ref_vec: FloatArray3 = -xaxis,
    pose_ang_lo: float = 45.0,
    pose_ang_hi: float = 135.0,
    diag_uv_lo: float = 0.15,
    diag_uv_hi: float = 0.9,
    ratio_uv_lo: float = 0.5,
    ratio_uv_hi: float = 2.0,
    grd_hgt_lo: float = 0.3,
    grd_plane_model: FloatArray4 = zxplane,
    disp_info: bool = True,
) -> BoolArrayN:
    """
    Filter multiple oriented bounding boxes based on the given conditions:
    obox's uv plane diagonal, aspect ratio, and uv plane normal vector alignment
    angle with the given ref_vec, the distance to the ground plane.

    Args:
        obox_candidates(type: list[o3d.BoundingBox])
        ref_vec=-xaxis (default) -- reference vector for obox pose alignment
        pose_ang_lo=np.pi/4 (default) -- obox's pose alignment lo bound angle
        pose_ang_hi=np.pi/4*3 (default) -- obox's pose alignment hi bound angle
        diag_uv_lo=0.15 (default) -- obox's uv plane diagonal lo bound
        diag_uv_hi=0.90 (default) -- obox's uv plane diagonal hi bound
        ratio_uv_lo=0.5 (default) -- obox's uv plane aspect ratio lo bound
        ratio_uv_hi=2.0 (default) -- obox's uv plane aspect ratio hi bound
        grd_hght_lo=0.3 (default) -- obox's center ground clearance lo bound
        grd_plane_model=zxplane (default) -- ground plane model
        disp_info=True (default) -- controls info display

    Returns:
        selected_index(type: list[bool]) -- the selected obox indicies

    """
    # Sanity check for the input arguments
    sck.is_valid_o3dobj_list(
        o3dobjs=obox_candidates,
        o3dobjs_name=nameof(obox_candidates),
        selected_idx=[True] * len(obox_candidates),
    )
    sck.is_nonzero_vector3(ref_vec, nameof(ref_vec))
    sck.is_float_hi_gt_lo(
        pose_ang_lo, pose_ang_hi, "pose_ang_lo", "pose_ang_hi"
    )
    sck.is_float_hi_gt_lo(diag_uv_lo, diag_uv_hi, "diag_uv_lo", "diag_uv_hi")
    sck.is_float_hi_gt_lo(
        ratio_uv_lo, ratio_uv_hi, "ratio_uv_lo", "ratio_uv_hi"
    )
    sck.is_positive_float(val=grd_hgt_lo, val_name=nameof(grd_hgt_lo))
    sck.is_valid_plane_model(grd_plane_model, nameof(grd_plane_model))

    # get necessary metrics for the screening in the next step
    ref_vec = ref_vec / np.linalg.norm(ref_vec + 1e-16)
    [aa, bb, cc, dd] = grd_plane_model
    num_candidates = len(obox_candidates)
    selected_index = np.repeat(False, num_candidates)
    ctrs, eu, ev, ew, uu, vv, ww = get_oriented_bounding_boxes_info(
        oboxes=obox_candidates
    )
    eduv = (eu**2 + ev**2) ** 0.5
    asuv = np.maximum(eu / ev, ev / eu)
    ang_ww2refvec = np.array(
        vector_angle(ww, np.tile(ref_vec, (ww.shape[0], 1))), dtype=np.float64
    )
    grd_hgt = (
        np.abs(aa * ctrs[:, 0] + bb * ctrs[:, 1] + cc * ctrs[:, 2] + dd)
        / (aa * aa + bb * bb + cc * cc) ** 0.5
    )

    selected_index = (
        ((ang_ww2refvec <= pose_ang_lo) | (ang_ww2refvec >= pose_ang_hi))
        & ((eduv >= diag_uv_lo) & (eduv <= diag_uv_hi))
        & ((asuv >= ratio_uv_lo) & (asuv <= ratio_uv_hi))
        & (grd_hgt >= grd_hgt_lo)
    )
    if disp_info:
        for jj, obox in enumerate(obox_candidates):
            print(
                f"[INFO: {currentFuncName()}]: "
                f" {jj}: "
                f"eu={eu[jj]:.2e}, ev={ev[jj]:.2e}, eduv={eduv[jj]:.2e}, "
                f"asuv={asuv[jj]:.2e}, ang_ww2refvec="
                f"{ang_ww2refvec[jj] / np.pi * 180.0:.1f}, "
                f"grd_hgt={grd_hgt[jj]:.2e}, selected={selected_index[jj]}"
            )

    return selected_index


def draw_planar_patches(
    oboxes: list[OrientedBoundingBox],
    o3dobj: PointCloud | TriangleMesh | OrientedBoundingBox | None = None,
    uvw_scale: float = 1.0,
    uvw_selected: BoolArray3 = np.array([True, True, True]),
    mu_color: FloatArray3 = CP.RED_DARK,
    mv_color: FloatArray3 = CP.GREEN_DARK,
    mw_color: FloatArray3 = CP.BLUE_DARK,
    disp_info: bool = True,
) -> None:
    """
    Draw planar patches with optional overlay on the given base
        o3d object (pcd, mesh or obox)

    Args:
        oboxes(type: list[o3d.BoundingBox]):
            the oboxes that represent the planar patches
        o3dobj(type: o3d.PointCloud | o3d.TriangleMesh):
            the o3d object to superimpose the planr pathes onto
        uvw_scale(type: float)=1.0 (default)
            -- set uvw arrow mesh scale
        uvw_selected(type: list[bool])=[True, True, True] (default)
            -- control which uvw arrow mesh to display
        mu_color(type: np.array([r, g, b]))=CP.RED_DARK (defalut)
        mv_color(type: np.array([r, g, b]))=CP.GREEN_DARK (defalut)
        mw_color(type: np.array([r, g, b]))=CP.BLUE_DARK (defalut)
            -- set uvw arrow mesh color
        disp_info(type: bool): controls info display, default is True

    Returns:
        None

    """
    # Sanity check for the input arguments
    sck.is_valid_o3dobj_list(
        o3dobjs=oboxes,
        o3dobjs_name=nameof(oboxes),
        selected_idx=[True] * len(oboxes),
    )
    sck.is_positive_float(uvw_scale, nameof(uvw_scale))
    sck.is_valid_color_vector3(mu_color, nameof(mu_color))
    sck.is_valid_color_vector3(mv_color, nameof(mv_color))
    sck.is_valid_color_vector3(mw_color, nameof(mw_color))

    if disp_info:
        print(
            f"[INFO: {currentFuncName()}]: "
            f"planar patches: #, dmx, dmn, ctr, nvec"
        )
    obox_objs = []
    obox_objs.append(get_xyz_axes(frame_size=0.5))
    if o3dobj is not None:
        obox_objs.append(o3dobj)
    for jj, obox in enumerate(oboxes):
        ctr = obox.get_center()
        uu, vv, ww = obox.R[:, 0], obox.R[:, 1], obox.R[:, 2]
        [eu, ev, ew] = obox.extent
        dmx, dmn = max(eu, ev), min(eu, ev)
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            obox, scale=[1, 1, 0.0001]
        )
        mesh.paint_uniform_color(obox.color)
        obox_objs.append(mesh)
        obox_objs.append(obox)
        if np.any(uvw_selected):
            mu = get_arrow_mesh(ctr, ctr + uu, uvw_scale, mu_color)
            mv = get_arrow_mesh(ctr, ctr + vv, uvw_scale, mv_color)
            mw = get_arrow_mesh(ctr, ctr + ww, uvw_scale, mw_color)
            uvw_mesh = np.sum(np.array([mu, mv, mw])[np.array(uvw_selected)])
            obox_objs.append(uvw_mesh)

        if disp_info:
            print(
                f"  #{jj:02d}: {dmx:.2e} | {dmn:.2e} | "
                f"[{ctr[0]:+.2e}, {ctr[1]:+.2e}, {ctr[2]:+.2e}] | "
                f"[{ww[0]:+.2e}, {ww[1]:+.2e}, {ww[2]:+.2e}]"
            )

    o3d.visualization.draw(obox_objs)


def spin_oriented_bounding_box(
    obox: OrientedBoundingBox, ref_vec: FloatArray3
) -> Tuple[OrientedBoundingBox, FloatArray3x3, FloatArray3]:
    """
    Spin a oriented bounding box along its ww vec so to align its uu or vv
        axis with a reference vector

    Args:
        obox(type: o3d.OrientedBoundingBox)
        ref_vec(type: np.ndarray[xn, yn, zn])

    Returns:
        obox(type: o3d.OrientedBoundingBox)

    """
    # rotate the obox along its ww-axis to align its upright vector
    # (either uu or vv depending on which one is closer to the ref_vec),
    # so the obox aligns better with the tri-axes
    uu, vv, _ = obox.R[:, 0], obox.R[:, 1], obox.R[:, 2]
    tu2ref_vec: float = np.dot(uu, ref_vec)
    tv2ref_vec: float = np.dot(vv, ref_vec)
    angw: float
    if np.abs(tu2ref_vec) >= np.abs(tv2ref_vec):
        angw = (float)(vector_angle(uu, ref_vec))
    else:
        angw = (float)(vector_angle(vv, ref_vec))
    rotmat: FloatArray3x3 = obox.get_rotation_matrix_from_xyz((0, 0, angw))
    octr: FloatArray3 = obox.get_center()
    obox_t: OrientedBoundingBox = (
        obox.translate(translation=-octr, relative=True)
        .rotate(R=rotmat, center=(0, 0, 0))
        .translate(translation=octr, relative=True)
    )

    return (obox_t, rotmat, octr)


def flip_obox(
    obox: OrientedBoundingBox, ref_vec: FloatArray3
) -> Tuple[OrientedBoundingBox, FloatArray3x3, FloatArray3]:
    """
    Flip a oriented bounding box if its ww vec pointing to -ref_vec

    Args:
        obox(type: o3d.OrientedBoundingBox)
        ref_vec(type: np.ndarray[xn, yn, zn])

    Returns:
        obox(type: o3d.OrientedBoundingBox)

    """
    rotmat: FloatArray3x3
    octr: FloatArray3 = obox.get_center()
    uu, vv, ww = obox.R[:, 0], obox.R[:, 1], obox.R[:, 2]
    ss = np.sign(np.dot(ww, ref_vec))
    ss = 1.0 if ss == 0.0 else ss
    if ss < 0.0:
        AA: FloatArray3x3 = np.array(obox.R)
        BB: FloatArray3x3 = np.array([uu, ss * vv, ss * ww]).T
        rotmat, _ = rigid_transform_3D(AA, BB)
        obox_t = (
            obox.translate(translation=-octr, relative=True)
            .rotate(R=rotmat, center=(0, 0, 0))
            .translate(translation=octr, relative=True)
        )
    else:
        rotmat = np.eye(3)
        obox_t = obox

    return (obox_t, rotmat, octr)


def draw_oriented_bounding_boxes(
    vis: o3d.cpu.pybind.visualization.O3DVisualizer,
    oboxes: list[OrientedBoundingBox],
    vec2sensor: FloatArray3,
    label_name: str,
    uvw_scale: float = 1.0,
    uvw_selected: BoolArray3 = np.array([True, True, True]),
    mu_color: FloatArray3 = CP.RED_DARK,
    mv_color: FloatArray3 = CP.GREEN_DARK,
    mw_color: FloatArray3 = CP.BLUE_DARK,
    ref_plane_model: FloatArray4 = zxplane,
    disp_info: bool = True,
):
    """
    Draw planar patches with optional overlay on the given point cloud

    Args:
        vis(type: o3d.visualization.O3DVisualizer)
        oboxes(type: list[o3d.OrientedBoundingBox])
        vec2sensor(type: np.ndarray[xn, yn, zn])
        label_name(type: str)
        uvw_scale(type: float)=1.0 (default)
            -- set uvw arrow mesh scale
        uvw_selected(type: BoolArray3)=[True, True, True] (default)
            -- control which uvw arrow mesh to display
        mu_color(type: np.array([r, g, b]))=CP.RED_DARK (defalut)
        mv_color(type: np.array([r, g, b]))=CP.GREEN_DARK (defalut)
        mw_color(type: np.array([r, g, b]))=CP.BLUE_DARK (defalut)
        ref_plane_model(type: FloatArray4)=zxplane (default)
        disp_info(type: boolean)=True (default)

    Returns:
        None

    """
    # Sanity check for the input arguments
    sck.is_valid_o3dobj_list(
        o3dobjs=oboxes,
        o3dobjs_name=nameof(oboxes),
        selected_idx=[True] * len(oboxes),
    )
    sck.is_positive_float(uvw_scale, nameof(uvw_scale))
    sck.is_valid_color_vector3(mu_color, nameof(mu_color))
    sck.is_valid_color_vector3(mv_color, nameof(mv_color))
    sck.is_valid_color_vector3(mw_color, nameof(mw_color))
    sck.is_valid_plane_model(ref_plane_model, "ref_plane_model")

    if disp_info:
        print(
            f"[INFO: {currentFuncName()}]: "
            f"scaling results: #, dmx, dmn, dist2ref, ctr, nvec"
        )
    [aa, bb, cc, dd] = ref_plane_model
    for jj, this_obox in enumerate(oboxes):
        obox, _, _ = flip_obox(this_obox, vec2sensor)
        # # flip obox if its normal vector points to -vec2sensor
        # ss = np.sign(np.dot(obox.R[:, 2], vec2sensor))
        # ss = 1.0 if ss == 0.0 else ss
        # if ss < 0.0:
        #     rr = obox.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        #     ov = obox.get_center()
        #     obox.translate(translation=-ov, relative=True).rotate(
        #         R=rr, center=(0, 0, 0)
        #     ).translate(translation=ov, relative=True)
        # get necessary metrics for later visualization
        uu, vv, ww = obox.R[:, 0], obox.R[:, 1], obox.R[:, 2]
        ctr = obox.get_center()
        ex, ey = obox.extent[0], obox.extent[1]
        dmx, dmn = max(ex, ey), min(ex, ey)
        dist2ref = (
            np.sum(aa * ctr[0] + bb * ctr[1] + cc * ctr[2] + dd)
            / (aa * aa + bb * bb + cc * cc) ** 0.5
        )
        # generate a mesh to visualize the obox
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            obox, scale=[1, 1, 0.0001]
        )
        mesh.paint_uniform_color(obox.color)

        # add the visualization elements
        vis.add_geometry(f"{label_name}_obox{jj:02d}", obox)
        if np.any(uvw_selected):
            mu = get_arrow_mesh(ctr, ctr + uu, uvw_scale, mu_color)
            mv = get_arrow_mesh(ctr, ctr + vv, uvw_scale, mv_color)
            mw = get_arrow_mesh(ctr, ctr + ww, uvw_scale, mw_color)
            uvw_mesh = np.sum(np.array([mu, mv, mw])[np.array(uvw_selected)])
            vis.add_geometry(f"{label_name}_mesh{jj:02d}", mesh + uvw_mesh)
        else:
            vis.add_geometry(f"{label_name}_mesh{jj:02d}", mesh)
        vis.add_3d_label(
            obox.get_center(),
            f"{jj}|({int(dmn * m2mm)}|{int(dmx * m2mm)}|"
            f"{int(dist2ref * m2mm)})",
        )

        if disp_info:
            print(
                f"  #{jj:02d}: {dmx:.2e} | {dmn:.2e} | {dist2ref:.2e} | "
                f"[{ctr[0]:+.2e}, {ctr[1]:+.2e}, {ctr[2]:+.2e}] | "
                f"[{ww[0]:+.2e}, {ww[1]:+.2e}, {ww[2]:+.2e}]"
            )
