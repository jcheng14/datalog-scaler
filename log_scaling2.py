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
    disp_info: bool = True

    data_type: list[str] = [
        "snoqualmie-phtgm",
        "snoqualmie-lidar",
        "inshop-phtgm",
        "inshop-lidar",
    ]
    which_load: int  # 1,2,3,4 for snoqualmie onsite load number
    whick_snr: int
    which_end: str  # 'frt' or 'bck'
    which_step: int
    vec2sky: FloatArray3
    vec2snr: FloatArray3
    ptg2real_scale: float = 1.0
    floor_ht: float = 1.5
    folder_name: str = ""
    mrun_name: str = ""
    obj_name: str = ""

    pcd_frt: PointCloud
    pcd_bck: PointCloud
    logfrt_name: str
    logbck_name: str
    logfrt_axes: FloatArray3x3
    logbck_axes: FloatArray3x3
    logfrt_ref_plane_axis_pos: float
    logbck_ref_plane_axis_pos: float
    # frt: the desired pcd tri-axes and translation vector in the global coord
    ptg2real_scale_frt: float = 1.0
    logfrt_axes = np.array([-ut.xaxis, -ut.yaxis, ut.zaxis]).T
    logfrt_ref_plane_axis_pos = 0.0
    # bck: the desired pcd tri-axes and translation vector in the global coord
    ptg2real_scale_bck: float = 1.0
    logbck_axes = np.array([ut.xaxis, ut.yaxis, ut.zaxis]).T
    logbck_ref_plane_axis_pos = 0.0
    dist_bt_refs_ft: float = 0.0

    ang_offset: float = np.pi / 6
    uvw_scale = 0.2

    which_data: str = data_type[2]
    match which_data:
        case "snoqualmie-phtgm":
            # for snoqualmie phtgm pcds on 5/23/2023
            vec2sky, vec2snr = -ut.yaxis, -ut.zaxis
            # which_load, which_end = 1, "frt"
            # ptg2real_scale, floor_ht = 0.568, 2.15
            # which_load, which_end = 1, "bck"
            # ptg2real_scale, floor_ht = 0.539, 1.7
            # which_load, which_end = 2, "frt"
            # ptg2real_scale, floor_ht = 1.027, 2.15
            # which_load, which_end = 2, "bck"
            # ptg2real_scale, floor_ht = 0.514, 1.7
            which_load, dist_bt_refs_ft = 3, 38.25
            ptg2real_scale_frt, ptg2real_scale_bck = 0.529, 1.844
            floor_ht_frt, floor_ht_bck = 2.15, 1.70
            # which_load, which_end = 3, "bck"
            # ptg2real_scale, floor_ht = 1.844, 1.7
            # which_load, which_end = 4, "frt"
            # ptg2real_scale, floor_ht = 0.529, 1.8
            # which_load, which_end = 4, "bck"
            # ptg2real_scale, floor_ht = 1.257, 1.7

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            logfrt_name = f"mrun_snoload{which_load}frt_finefeat"
            logbck_name = f"mrun_snoload{which_load}bck_finefeat"
            logfrt_ref_plane_axis_pos = 0.0
            logbck_ref_plane_axis_pos = (
                dist_bt_refs_ft * ut.foot2inch * ut.inch2mm / ut.m2mm
            )

        case "snoqualmie-lidar":
            # for snoqualmie lidar pcds on 5/23/2023
            vec2sky, vec2snr = -ut.yaxis, -ut.xaxis
            which_load, which_snr, which_step = 3, 102, 1
            ptg2real_scale_frt, ptg2real_scale_bck = 0.529, 1.844
            floor_ht_frt, floor_ht_bck = 2.15, 1.7
            dist_bt_refs_ft = 38.25

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            logfrt_name = f"pcd-{which_load}-{which_snr}-front-{which_step}"
            logbck_name = f"pcd-{which_load}-{which_snr}-back-{which_step}"
            logfrt_ref_plane_axis_pos = 0.0
            logbck_ref_plane_axis_pos = (
                dist_bt_refs_ft * ut.foot2inch * ut.inch2mm / ut.m2mm
            )

        case "inshop-phtgm":
            vec2sky, vec2snr = -ut.yaxis, -ut.zaxis
            folder_name = "C:\\Users\\SteveYin\\MyCode"
            mrun_name = "datalog-mvp-scaler\\data_local"
            # Flir camera, multiple captures
            logfrt_name = "mrun_logs_on_testbed_20MP_202304251740_18"  # front
            logbck_name = "mrun_logs_on_testbed_20MP_202304271449_9"  # back
            logfrt_ref_plane_axis_pos = 0
            logbck_ref_plane_axis_pos = 1480.0

        case "inshop-lidar":
            vec2sky, vec2snr = ut.zaxis, -ut.xaxis
            folder_name = "C:\\Users\\SteveYin\\MyCode"
            mrun_name = "datalog-mvp-scaler\\data_local"
            # Livox MID-70, multiple captures
            logfrt_name = "pcd-apr25-in-5ftwd-3.4166ftht-1_crop.ply"
            logbck_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-1_crop.ply"
            # logfrt_name = "pcd-apr25-in-5ftwd-3.4166ftht-2_crop.ply"
            # logbck_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-2_crop.ply"
            logfrt_ref_plane_axis_pos = 0
            logbck_ref_plane_axis_pos = 1480.0

        case _:
            sys.exit("case not handled when loading pcd data file...")

    # load front section pcd from the file
    pcd_frt, _ = ut.load_dataset(
        data_path=f"{folder_name}\\{mrun_name}\\{logfrt_name}_crop.ply"
    )
    ind: list[int]
    # a little denoise to sparse the PCD
    if which_data == "inshop-phtgm":
        pcd_frt, ind = pcd_frt.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=0.5
        )
    # sanity check before proceeding further
    ut.sck.is_valid_pcd(pcd_frt, nameof(pcd_frt))
    pcd_frt.scale(ptg2real_scale_frt, pcd_frt.get_center())

    # load back section pcd from the file
    pcd_bck, _ = ut.load_dataset(
        data_path=f"{folder_name}\\{mrun_name}\\{logbck_name}_crop.ply"
    )
    # a little denoise to sparse the PCD
    if which_data == "inshop-phtgm":
        pcd_bck, ind = pcd_bck.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=0.5
        )
    # sanity check before proceesing further
    ut.sck.is_valid_pcd(pcd_bck, nameof(pcd_bck))
    pcd_bck.scale(ptg2real_scale_bck, pcd_bck.get_center())

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
        vec2snr=vec2snr,
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
        vec2snr=vec2snr,
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
    match which_data:
        # dealing with photogrammetry specific params
        case "inshop_phtgm":
            param_patch_det = ParamPlanarPatchDetection(
                normal_variance_threshold_deg=30,  # 60
                coplanarity_deg=75,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=0,  # 0
                search_knn=30,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ParamFiducialDetection(
                fid_patch_ang_lo=np.pi / 4,
                fid_patch_ang_hi=np.pi / 4 * 3,
                fid_patch_ratio_uv_lo=0.33,
                fid_patch_ratio_uv_hi=3.0,
            )
            # setup log patch detection parameters
            param_logend_det = ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.12,
                diag_uv_hi=0.72,
                ratio_uv_lo=0.75,
                ratio_uv_hi=1.25,
                grd_hgt_lo=0.12,
            )

        case "snoqualmie-phtgm":
            param_patch_det = ParamPlanarPatchDetection(
                normal_variance_threshold_deg=25,  # 60
                coplanarity_deg=60,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=10,  # 0
                search_knn=5,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ParamFiducialDetection(
                fid_patch_ang_lo=np.pi / 6,
                fid_patch_ang_hi=np.pi / 6 * 5,
                fid_patch_ratio_uv_lo=0.25,
                fid_patch_ratio_uv_hi=4.0,
            )
            # setup log patch detection parameters
            param_logend_det = ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.05,
                diag_uv_hi=0.75,
                ratio_uv_lo=0.55,
                ratio_uv_hi=1.82,
                grd_hgt_lo=1.7,
            )

        case "snoqualmie-lidar":
            param_patch_det = ParamPlanarPatchDetection(
                normal_variance_threshold_deg=25,  # 60
                coplanarity_deg=60,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=10,  # 0
                search_knn=5,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ParamFiducialDetection(
                fid_patch_ang_lo=np.pi / 6,
                fid_patch_ang_hi=np.pi / 6 * 5,
                fid_patch_ratio_uv_lo=0.25,
                fid_patch_ratio_uv_hi=4.0,
            )
            # setup log patch detection parameters
            param_logend_det = ParamLogendDetection(
                pose_ang_lo=ang_offset,
                pose_ang_hi=np.pi - ang_offset,
                diag_uv_lo=0.05,
                diag_uv_hi=0.75,
                ratio_uv_lo=0.55,
                ratio_uv_hi=1.82,
                grd_hgt_lo=1.7,
            )

        # dealing with lidar pcd, using Lidar specific params
        case "inshop-lidar":
            param_patch_det = ParamPlanarPatchDetection(
                normal_variance_threshold_deg=45,  # 60
                coplanarity_deg=45,  # 75
                outlier_ratio=0.25,  # 0.75
                min_plane_edge_len=0,  # 0
                min_num_pts=0,  # 0
                search_knn=10,  # 30
            )
            # setup ref_plane/fiducials detection parameters
            param_fid_det = ParamFiducialDetection(
                fid_patch_ang_lo=ang_offset,
                fid_patch_ang_hi=np.pi - ang_offset,
                fid_patch_ratio_uv_lo=0.33,
                fid_patch_ratio_uv_hi=3.0,
            )
            # setup log patch detection parameters
            param_logend_det = ParamLogendDetection(
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
                f"which_data definition {which_data} not supported."
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
        vec2snr=logfrt_axes[:, 1],
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
        vec2snr=logbck_axes[:, 1],
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
        uvw_scale=uvw_scale,
        uvw_selected=np.array([False, False, True]),
    )
    logbck.draw_log_scaling_results(
        vis=vis,
        label_name="bck",
        uvw_scale=uvw_scale,
        uvw_selected=np.array([False, False, True]),
    )
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

else:
    sys.exit(f"[INFO: {ut.currentFuncName()}]: nothing to show, check code!")
