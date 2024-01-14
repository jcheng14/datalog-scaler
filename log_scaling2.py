# ----------------------------------------------------------------------------
# - Pointcloud Data Processing for log-end segmentation & measurement        -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
import sys
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from varname import nameof
import utility as ut
from utility import FloatArray3, FloatArray4, FloatArrayN, IntArrayN  # noqa
from utility import FloatArray3x3, FloatArrayNx3, BoolArray3, BoolArrayN  # noqa
from utility import PointCloud, TriangleMesh, OrientedBoundingBox  # noqa
from utility import (
    ParamPlanarPatchDetection,
    ParamFiducialDetection,
    ParamLogendDetection,
)
from pointcloudhandler import PointcloudHandler


if __name__ == "__main__":
    """
    The main function conducts an automated log scaling process:
        1) load PCD file (support both TriangleMesh and PointCloud data formats)
        2) preprocessing the PCD -- ground plane detection to get the rotmat
            for rotating the pcd
        3) put the transformed PCD in a global coord, create a new pcd
        3) re-detect the ground plan, detect planar patches, and detect
            a reference plane from the new PCD
        4) run segment selection to find individual log-end in the PCD
        5) post-processing to obtain log-end diameter and depth
        6) visualization
    """
    # =======================================================================
    # STEP 1: load PCD files
    # =======================================================================
    # setup condition parameters
    disp_progress: bool = True
    disp_info: bool = False
    phtgm_or_lidar: bool = True
    opt_clustering: list[str] = [
        "PHTGM_SHOP",
        "PHTGM_SNO",
        "LIDAR_SHOP",
    ]
    use_clustering: str = opt_clustering[1]

    # setup PCD file path
    pcd_folder: str
    logfrt_name: str
    logbck_name: str

    # setup log scaling coordinate parameters
    vec2sky: FloatArray3
    vec2sensor: FloatArray3
    ang_offset: float = np.pi / 6
    uvw_mesh_scale: float = 0.2
    # frt: the desired pcd tri-axes and translation vector in the global coord
    logfrt_axes: FloatArray3x3 = np.array([-ut.xaxis, -ut.yaxis, ut.zaxis]).T
    phtgm2real_scale_frt: float = 1.0
    logfrt_ref_plane_axis_pos: float = 0.0
    # bck: the desired pcd tri-axes and translation vector in the global coord
    logbck_axes: FloatArray3x3 = np.array([ut.xaxis, ut.yaxis, ut.zaxis]).T
    phtgm2real_scale_bck: float = 1.0
    logbck_ref_plane_axis_pos: float = 0.0

    match use_clustering:
        # Flir camera, multiple captures
        case "PHTGM_SHOP":
            vec2sky = -ut.yaxis
            vec2sensor = -ut.zaxis
            pcd_folder = (
                "C:\\Users\\SteveYin\\MyCode\\datalog-mvp-scaler\\data_local"
            )
            logfrt_name = "mrun_logs_on_testbed_20MP_202304251740_18_crop.ply"
            logbck_name = "mrun_logs_on_testbed_20MP_202304271449_9_crop.ply"
            logfrt_ref_plane_axis_pos = 0
            logbck_ref_plane_axis_pos = 1480.0

        case "PHTGM_SNO":
            vec2sky = -ut.yaxis
            vec2sensor = -ut.zaxis
            pcd_folder = "C:\\Users\\SteveYin\\MyData\\snofield20230523"
            logfrt_name = "mrun_snoload3frt_finefeat_crop.ply"
            logbck_name = "mrun_snoload3bck_finefeat_crop.ply"
            logfrt_ref_plane_axis_pos = 0.0
            logbck_ref_plane_axis_pos = (
                38.25 * ut.foot2inch * ut.inch2mm / ut.m2mm
            )
            phtgm2real_scale_frt = 0.529
            phtgm2real_scale_bck = 1.844

        case "LIDAR_SHOP":
            vec2sky = ut.zaxis
            vec2sensor = -ut.xaxis
            pcd_folder = "C:\\Users\\SteveYin\\MyCode\\datalog-mvp-scaler\\data"
            # Livox MID-70, multiple captures
            logfrt_name = "pcd-apr25-in-5ftwd-3.4166ftht-1_crop.ply"
            logbck_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-1_crop.ply"
            # logfrt_name = "pcd-apr25-in-5ftwd-3.4166ftht-2_crop.ply"
            # logbck_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-2_crop.ply"
            logfrt_ref_plane_axis_pos = 0
            logbck_ref_plane_axis_pos = 1480.0

        case _:
            sys.exit("case not handled")

    # load front section pcd from the file
    pcd_frt: PointCloud = ut.load_pcd(pcd_path=f"{pcd_folder}\\{logfrt_name}")
    ind: list[int]
    # a little denoise to sparse the PCD
    # if phtgm_or_lidar:
    #     pcd_frt, ind = pcd_frt.remove_statistical_outlier(
    #         nb_neighbors=30, std_ratio=0.5
    #     )
    # else:
    #     pcd_frt, ind = pcd_frt.remove_statistical_outlier(
    #         nb_neighbors=10, std_ratio=0.5
    #     )
    # sanity check before proceeding further
    ut.sck.is_valid_pcd(pcd_frt, nameof(pcd_frt))
    pcd_frt.scale(phtgm2real_scale_frt, pcd_frt.get_center())

    # load back section pcd from the file
    pcd_bck: PointCloud = ut.load_pcd(pcd_path=f"{pcd_folder}\\{logbck_name}")
    # a little denoise to sparse the PCD
    # if phtgm_or_lidar:
    #     pcd_bck, ind = pcd_bck.remove_statistical_outlier(
    #         nb_neighbors=30, std_ratio=0.5
    #     )
    # else:
    #     pcd_bck, ind = pcd_bck.remove_statistical_outlier(
    #         nb_neighbors=10, std_ratio=0.5
    #     )
    # sanity check before proceesing further
    ut.sck.is_valid_pcd(pcd_bck, nameof(pcd_bck))
    pcd_bck.scale(phtgm2real_scale_bck, pcd_bck.get_center())

    # =======================================================================
    # STEP 2: preprocessing PCD
    # find a ground plane, use the ground plane pose to fix the PCD alignment
    # with global axes, z-up, x-lateral, sensor looking at +y or -y
    # =======================================================================
    # instantiate a PointcloundHandler instance for logfrt
    voi_frt: PointcloudHandler = PointcloudHandler(
        pcd_raw=pcd_frt,
        pcd_idx=[-1] * len(pcd_frt.points),
        vec2sky=vec2sky,
        vec2sensor=vec2sensor,
        disp_info=disp_info,
        disp_progress=disp_progress,
    )
    logfrt_pcd: PointCloud = voi_frt.preprocess_pcd(
        dst_pcd_axes=logfrt_axes,
        ang_offset=ang_offset,
        max_num_planes=200,
    )
    ctr = logfrt_pcd.get_center()
    rotmat = logfrt_pcd.get_rotation_matrix_from_xyz([0, 0, np.pi])
    logfrt_pcd.translate(translation=-ctr, relative=True).rotate(
        R=rotmat, center=(0, 0, 0)
    ).translate(translation=ctr, relative=True)

    # instantiate a PointcloundHandler instance for logbck
    voi_bck: PointcloudHandler = PointcloudHandler(
        pcd_raw=pcd_bck,
        pcd_idx=[-1] * len(pcd_bck.points),
        vec2sky=vec2sky,
        vec2sensor=vec2sensor,
        disp_info=disp_info,
        disp_progress=disp_progress,
    )
    logbck_pcd: PointCloud = voi_bck.preprocess_pcd(
        dst_pcd_axes=logbck_axes,
        ang_offset=ang_offset,
        max_num_planes=200,
    )
    ctr = logbck_pcd.get_center()
    rotmat = logbck_pcd.get_rotation_matrix_from_xyz([0, 0, np.pi])
    logbck_pcd.translate(translation=-ctr, relative=True).rotate(
        R=rotmat, center=(0, 0, 0)
    ).translate(translation=ctr, relative=True)

    # =======================================================================
    # STEP 3: define parameters for the segmentation and depth measurements
    # =======================================================================
    param_patch_det: ParamPlanarPatchDetection
    param_fid_det: ParamFiducialDetection
    param_logend_det: ParamLogendDetection
    match use_clustering:
        # dealing with photogrammetry specific params
        case "PHTGM_SHOP":
            param_patch_det = ut.ParamPlanarPatchDetection(
                normal_variance_threshold_deg=30,  # 60
                coplanarity_deg=75,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=0,  # 0
                search_knn=30,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ut.ParamFiducialDetection(
                fid_patch_ang_lo=ang_offset,
                fid_patch_ang_hi=np.pi - ang_offset,
                fid_patch_ratio_uv_lo=0.33,
                fid_patch_ratio_uv_hi=3.0,
            )
            # setup log-end patch detection parameters
            param_logend_det = ut.ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.12,
                diag_uv_hi=0.72,
                ratio_uv_lo=0.75,
                ratio_uv_hi=1.25,
                grd_hgt_lo=0.12,
            )

        case "PHTGM_SNO":
            param_patch_det = ut.ParamPlanarPatchDetection(
                normal_variance_threshold_deg=20,  # 60
                coplanarity_deg=60,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=0,  # 0
                search_knn=30,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ut.ParamFiducialDetection(
                fid_patch_ang_lo=ang_offset,
                fid_patch_ang_hi=np.pi - ang_offset,
                fid_patch_ratio_uv_lo=0.33,
                fid_patch_ratio_uv_hi=3.0,
            )
            # setup log-end patch detection parameters
            param_logend_det = ut.ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.08,
                diag_uv_hi=1.0,
                ratio_uv_lo=0.75,
                ratio_uv_hi=1.33,
                grd_hgt_lo=1.0,
            )

        # dealing with lidar pcd, using Lidar specific params
        case "LIDAR_SHOP":
            param_patch_det = ut.ParamPlanarPatchDetection(
                normal_variance_threshold_deg=45,  # 60
                coplanarity_deg=45,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=0,  # 0
                search_knn=10,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ut.ParamFiducialDetection(
                fid_patch_ang_lo=ang_offset,
                fid_patch_ang_hi=np.pi - ang_offset,
                fid_patch_ratio_uv_lo=0.25,
                fid_patch_ratio_uv_hi=4.0,
            )
            # setup log patch detection parameters
            param_logend_det = ut.ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.08,
                diag_uv_hi=1.2,
                ratio_uv_lo=0.5,
                ratio_uv_hi=2.0,
                grd_hgt_lo=0.30,
            )

        # Unknown handling
        case _:
            sys.exit(
                f"[INFO: {ut.currentFuncName()}]: "
                f"use_clustering definition not supported in "
                f"opt_clustering({opt_clustering})"
            )

    # =======================================================================
    # STEP 4: do segmentation on PCD
    # do planar patch detection to find each individual log end surface
    # find a reference plane and its pose for log depth measurements
    # =======================================================================
    # logend front section scaling
    logfrt: PointcloudHandler = PointcloudHandler(
        pcd_raw=logfrt_pcd,
        pcd_idx=voi_frt.pcd_idx,
        vec2sky=logfrt_axes[:, 2],
        vec2sensor=logfrt_axes[:, 1],
        disp_info=disp_info,
        disp_progress=disp_progress,
    )
    logfrt.logend_scaling_prep(
        param_patch_det=param_patch_det,
        param_fid_det=param_fid_det,
        ang_offset=ang_offset,
        max_num_planes=200,
    )
    logfrt.logend_scaling(
        param_patch_det=param_patch_det,
        param_logend_det=param_logend_det,
    )

    # logend back section scaling
    logbck: PointcloudHandler = PointcloudHandler(
        pcd_raw=logbck_pcd,
        pcd_idx=voi_bck.pcd_idx,
        vec2sky=logbck_axes[:, 2],
        vec2sensor=logbck_axes[:, 1],
        disp_info=disp_info,
        disp_progress=disp_progress,
    )
    logbck.logend_scaling_prep(
        param_patch_det=param_patch_det,
        param_fid_det=param_fid_det,
        ang_offset=ang_offset,
        max_num_planes=200,
    )
    logbck.logend_scaling(
        param_patch_det=param_patch_det,
        param_logend_det=param_logend_det,
    )

    # setup visualization
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("log scaling", 1280, 720)
    vis.show_settings = True
    logfrt.draw_log_scaling_results(
        vis=vis,
        label_name="frt",
        uvw_scale=uvw_mesh_scale,
        uvw_selected=np.array([False, False, True]),
    )
    logbck.draw_log_scaling_results(
        vis=vis,
        label_name="bck",
        uvw_scale=uvw_mesh_scale,
        uvw_selected=np.array([False, False, True]),
    )
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

else:
    sys.exit(f"[INFO: {ut.currentFuncName()}]: nothing to show, check code!")
