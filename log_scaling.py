# ----------------------------------------------------------------------------
# - Pointcloud Data Processing for log-end segmentation & measurement        -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
import sys
import open3d as o3d
import open3d.visualization.gui as gui  # noqa
import numpy as np
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
        2) preprocessing the PCD
        3) detect ground plan and reference plane from the PCD
        4) run planar patch detector to segment individual log-end in the PCD
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
            which_load, which_end = 3, "frt"
            ptg2real_scale, floor_ht = 0.529, 2.15
            # which_load, which_end = 3, "bck"
            # ptg2real_scale, floor_ht = 1.844, 1.7
            # which_load, which_end = 4, "frt"
            # ptg2real_scale, floor_ht = 0.529, 1.8
            # which_load, which_end = 4, "bck"
            # ptg2real_scale, floor_ht = 1.257, 1.7

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"mrun_snoload{which_load}{which_end}_finefeat"

        case "snoqualmie-lidar":
            # for snoqualmie lidar pcds on 5/23/2023
            vec2sky, vec2snr = -ut.yaxis, -ut.xaxis
            which_load, which_snr, which_end = 3, 102, "back"
            which_step, floor_ht = 1, 1.7

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"pcd-{which_load}-{which_snr}-{which_end}-{which_step}"

        case "inshop-phtgm":
            vec2sky, vec2snr = -ut.yaxis, -ut.zaxis

            folder_name = "/home/jamesxyye/Downloads/freshconsulting-datalog-mvp-scaler-0752cd72c8cb/"
            mrun_name = "mrun"
            # Flir camera, multiple captures
            # obj_name = "mrun_logs_on_testbed_20MP_202304251740_18"  # front
            # obj_name = "mrun_logs_on_testbed_20MP_202304251740_5"  # front
            # obj_name = "mrun_logs_on_testbed_20MP_202304271449_9"  # back
            # Canon EOS 90D, logs and tubes
            # obj_name = "mrun_logs_on_testbed_32MP_202303291044"
            # obj_name = "mrun_tubes_on_testbed_32MP_202303271646"
            # iPhone 14PM recessed log close-up
            # obj_name = "mrun_logs_on_testbed_12MP_202304111609_8"

        case "inshop-lidar":
            vec2sky, vec2snr = ut.zaxis, -ut.xaxis
            folder_name = "C:\\Users\\SteveYin\\MyCode"
            mrun_name = "datalog-mvp-scaler\\data_local"
            # obj_name = "lidar_data_apr_12_1frame"
            # obj_name = "lidar_data_apr12_5ftwd_2fttall_1_frame"
            obj_name = "pcd-apr25-in-5ftwd-3.4166ftht-1"
            # obj_name = "pcd-apr25-in-5ftwd-3.4166ftht-2"
            # obj_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-1"
            # obj_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-2"
            # obj_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-1"
            # obj_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-2"
            # obj_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-1"
            # obj_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-2"

        case _:
            sys.exit("case not handled when loading pcd data file...")

    # load the pcd data
    pcd, _ = ut.load_dataset(
        data_path=f"{folder_name}/cropped_1.ply"
    )
    ut.sck.is_valid_pcd(pcd, "pcd")
    # a little denoise to sparse the inshop phtgm dense pcd
    if which_data == data_type[2]:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=0.5
        )
    pcd.scale(ptg2real_scale, pcd.get_center())
    pcd.estimate_normals()

    # =======================================================================
    # STEP 2: preprocessing PCD
    # find a ground plane, use the ground plane pose to fix the PCD alignment
    # with global axes, z-up, x-lateral, sensor looking at +y or -y
    # =======================================================================
    ang_offset: float = np.pi / 6
    uvw_scale: float = 0.2
    # the desired pcd tri-axes for placing the pcd in the global coord
    dst_pcd_axes: FloatArray3x3
    dst_pcd_axes = np.array([-ut.xaxis, -ut.yaxis, ut.zaxis]).T
    # dst_pcd_axes = np.array([ut.xaxis, ut.yaxis, ut.zaxis]).T

    # instantiate a PointcloundHandler instance
    voi: PointcloudHandler = PointcloudHandler(
        pcd_raw=pcd,
        pcd_idx=[-1] * len(pcd.points),
        vec2sky=vec2sky,
        vec2snr=vec2snr,
        disp_info=disp_info,
        disp_progress=disp_progress,
    )
    if disp_info:
        o3d.visualization.draw([voi.pcd, ut.get_xyz_axes(frame_size=0.5)])

    pcd_t: PointCloud = voi.preprocess_pcd(
        dst_pcd_axes=dst_pcd_axes,
        ang_offset=ang_offset,
        max_num_planes=200,
    )
    ctr: FloatArray3 = pcd_t.get_center()
    rotmat: FloatArray3x3 = pcd_t.get_rotation_matrix_from_xyz([0, 0, np.pi])
    pcd_t.translate(translation=-ctr, relative=True).rotate(
        R=rotmat, center=(0, 0, 0)
    ).translate(translation=ctr, relative=True)

    # test of no-rotation
    # pcd_t = pcd
    # dst_pcd_axes = np.array([np.cross(vec2sky, vec2snr), vec2snr, vec2sky]).T

    # =======================================================================
    # STEP 3: define parameters for the segmentation and depth measurements
    # =======================================================================
    param_patch_det: ParamPlanarPatchDetection
    param_fid_det: ParamFiducialDetection
    param_logend_det: ParamLogendDetection
    match which_data:
        # dealing with photogrammetry specific params
        case "inshop-phtgm":
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
            if which_end == "frt":
                param_patch_det = ParamPlanarPatchDetection(
                    normal_variance_threshold_deg=25,  # 60
                    coplanarity_deg=60,  # 75
                    outlier_ratio=0.25,  # 0.75
                    min_plane_edge_len=0,  # 0
                    min_num_pts=10,  # 0
                    search_knn=5,  # 30
                )
            else:
                param_patch_det = ParamPlanarPatchDetection(
                    normal_variance_threshold_deg=25,  # 60
                    coplanarity_deg=50,  # 75
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
            if which_end == "frt":
                param_patch_det = ParamPlanarPatchDetection(
                    normal_variance_threshold_deg=25,  # 60
                    coplanarity_deg=60,  # 75
                    outlier_ratio=0.25,  # 0.75
                    min_plane_edge_len=0,  # 0
                    min_num_pts=10,  # 0
                    search_knn=5,  # 30
                )
            else:
                param_patch_det = ParamPlanarPatchDetection(
                    normal_variance_threshold_deg=25,  # 60
                    coplanarity_deg=50,  # 75
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
    # instantiate a PointcloundHandler instance
    logsec: PointcloudHandler = PointcloudHandler(
        pcd_raw=pcd_t,
        pcd_idx=voi.pcd_idx,
        vec2sky=dst_pcd_axes[:, 2],
        vec2snr=dst_pcd_axes[:, 1],
        disp_info=disp_info,
        disp_progress=disp_progress,
    )

    logsec.logend_scaling_prep(
        param_patch_det=param_patch_det,
        param_fid_det=param_fid_det,
        ang_offset=ang_offset,
        max_num_planes=200,
        ref_plane_axis_position=0.0,
    )

    logsec.logend_scaling(
        param_patch_det=param_patch_det,
        param_logend_det=param_logend_det,
    )

    # setup visualization
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("log scaling", 1280, 720)
    vis.show_settings = True
    logsec.draw_log_scaling_results(
        vis=vis,
        label_name="log",
        uvw_scale=uvw_scale,
        uvw_selected=np.array([False, False, True]),
    )
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

else:
    print(f"[INFO: {ut.currentFuncName()}]: nothing to show, check code!")
