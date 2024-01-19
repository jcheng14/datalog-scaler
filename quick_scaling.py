# ----------------------------------------------------------------------------
# - Quick log scaling script for Snoqualmie onsite test datasets             -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
import sys
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utility as ut
import copy
from utility import FloatArray3, FloatArray4, FloatArrayN, IntArrayN  # noqa
from utility import FloatArray3x3, FloatArrayNx3, BoolArray3, BoolArrayN  # noqa
from utility import PointCloud, TriangleMesh, OrientedBoundingBox  # noqa
from pointcloudhandler import PointcloudHandler


if __name__ == "__main__":
    data_type: list[str] = [
        "snoqualmie-phtgm",
        "snoqualmie-lidar",
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
            # Snoqualmie test site data 5/23/2023
            # for snoqualmie phtgm pcds
            which_load, which_end, ptg2real_scale, floor_ht = (
                1,
                "frt",
                0.568,
                2.15,
            )
            # which_load, which_end, ptg2real_scale, floor_ht = 1, "bck", 0.539, 1.7
            # which_load, which_end, ptg2real_scale, floor_ht = 2, "frt", 1.027, 2.15
            # which_load, which_end, ptg2real_scale, floor_ht = 2, "bck", 0.514, 1.7
            # which_load, which_end, ptg2real_scale, floor_ht = 3, "frt", 0.529, 2.15
            # which_load, which_end, ptg2real_scale, floor_ht = 3, "bck", 1.844, 1.7
            # which_load, which_end, ptg2real_scale, floor_ht = 4, "frt", 0.529, 1.8
            # which_load, which_end, ptg2real_scale, floor_ht = (
            #     4,
            #     "bck",
            #     1.257,
            #     1.7,
            # )

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"mrun_snoload{which_load}{which_end}_finefeat"
            vec2sky, vec2snr = -ut.yaxis, -ut.zaxis

        case "snoqualmie-lidar":
            # Snoqualmie test site data 5/23/2023
            which_load, which_snr, which_end, which_step, floor_ht = (
                3,
                102,
                "back",
                1,
                1.7,
            )

            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"pcd-{which_load}-{which_snr}-{which_end}-{which_step}"
            vec2sky, vec2snr = -ut.yaxis, -ut.xaxis

        case _:
            sys.exit("case not handled when loading pcd data file...")

    # load the pcd data
    pcd, _ = ut.load_dataset(
        data_path=f"{folder_name}\\{mrun_name}\\{obj_name}_crop.ply"
    )
    ut.sck.is_valid_pcd(pcd, "pcd")
    pcd.scale(ptg2real_scale, pcd.get_center())
    pcd.estimate_normals()

    # instantiate a PointcloundHandler instance
    voi = PointcloudHandler(
        pcd_raw=pcd,
        pcd_idx=[-1] * len(pcd.points),
        vec2sky=vec2sky,
        vec2snr=vec2snr,
        disp_info=True,
        disp_progress=True,
    )

    # run quick log scaling
    # get snr_plane
    voi.get_snr_plane()

    # run ransac plane detection for the grd_plane detection
    voi.plane_ransac(
        npt_ransac=3,
        distance_threshold=0.01,
        num_ransac_iters=500,
        max_num_planes=200,
    )
    # get grd_plane
    voi.get_grd_plane(
        pose_ang_lo=np.pi / 12,
        pose_ang_hi=np.pi / 12 * 11,
    )

    # run planar patch detection for the ref_plane detection
    voi.planar_patches(
        the_pcd=voi.pcd,
        normal_variance_threshold_deg=25,  # noqa
        coplanarity_deg=75,
        outlier_ratio=0.25,
        min_plane_edge_length=0,
        min_num_points=0,
        search_knn=10,
    )
    ut.draw_planar_patches(
        oboxes=voi.planar_patch_oboxes,
        o3dobj=voi.pcd,
        uvw_scale=0.2,
        uvw_selected=np.array([True, True, True]),
        disp_info=True,
    )
    # get ref_plane
    voi.get_ref_plane(
        pose_ang_lo=np.pi / 6,
        pose_ang_hi=np.pi / 6 * 5,
        ratio_uv_lo=0.33,
        ratio_uv_hi=3.0,
    )

    # select points whose normal vecs align with vec2snr direction
    aa = np.array(voi.pcd.normals)
    bb = np.tile(vec2snr, (len(aa), 1))
    dd = ut.vector_angle(aa, bb)
    dd_idx = np.where((dd <= np.pi / 6) | (dd >= np.pi / 6 * 5))[0].tolist()
    alabel = 0
    voi.set_pcd_idx_label(selected_idx=dd_idx, label=alabel)
    idx_chosen = np.where(np.array(voi.pcd_idx) == alabel)[0].tolist()
    voi.set_pcd_color(idx_chosen, ut.CP.ORANGE)
    cl = pcd.select_by_index(idx_chosen)
    o3d.visualization.draw(
        [
            {"name": "pcd", "geometry": voi.pcd},
        ],
        show_ui=True,
        # point_size=5,
    )

    voi.planar_patches(
        the_pcd=cl,
        normal_variance_threshold_deg=25.0,
        coplanarity_deg=60.0,
        outlier_ratio=0.25,
        min_plane_edge_length=0.0,
        min_num_points=10,
        search_knn=5,
    )

    voi.filter_planar_patches(
        obox_candidates=voi.planar_patch_oboxes,
        pose_ang_lo=np.pi / 6,
        pose_ang_hi=np.pi / 6 * 5,
        diag_uv_lo=0.05,
        diag_uv_hi=0.75,
        ratio_uv_lo=0.55,
        ratio_uv_hi=1.82,
        grd_hgt_lo=floor_ht,
    )
    voi.get_log_scaling_results()

    # setup visualization
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("log scaling", 1280, 720)
    vis.show_settings = True
    voi.draw_log_scaling_results(
        vis=vis,
        label_name="log",
        uvw_scale=0.2,
        uvw_selected=np.array([False, False, True]),
    )
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
