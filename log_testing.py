# ----------------------------------------------------------------------------
# - Pointcloud Data testing and visualization                                -
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
from pointcloudhandler import PointcloudHandler


if __name__ == "__main__":
    data_type: list[str] = [
        "in-house-phtgm-mesh",
        "in-house-lidar",
        "snoqualmie-phtgm",
        "snoqualmie-lidar",
    ]
    which_data: str = data_type[2]
    which_load: int  # 1,2,3,4 for snoqualmie onsite load number
    whick_snr: int
    which_end: str  # 'frt' or 'bck'
    which_step: int
    ptg2real_scale: float = 1.0
    grd_hgt_lo: float = 1.5
    folder_name: str = ""
    mrun_name: str = ""
    obj_name: str = ""

    # for snoqualmie phtgm pcds
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 1, "frt", 0.568, 2.15
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 1, "bck", 0.539, 1.7
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 2, "frt", 1.027, 2.15
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 2, "bck", 0.514, 1.7
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 3, "frt", 0.529, 2.15
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 3, "bck", 1.844, 1.7
    # which_load, which_end, ptg2real_scale, grd_hgt_lo = 4, "frt", 0.529, 1.8
    which_load, which_end, ptg2real_scale, grd_hgt_lo = 4, "bck", 1.257, 1.7

    # for snoqualmie lidar pcds
    # which_load, which_snr, which_end, which_step, grd_hgt_lo = (
    #     3,
    #     102,
    #     "back",
    #     1,
    #     1.7,
    # )  # noqa

    test_case: list[str] = [
        "pcd_handler",
        "pcd_color_pick",
        "pcd_idx_label",
        "pcd_normals_pick",
    ]
    which_test: str = test_case[3]

    match which_data:
        case "in-house-phtgm-mesh":
            # Flir camera, multiple captures
            folder_name = "C:\\Users\\SteveYin\\MyCode\\datalog-photogrammetry"
            obj_name = "texturedMesh.obj"
            mrun_name = "mrun_logs_on_testbed_20MP_202304251740_18"

            pcd_folder: str = (
                "C:\\Users\\SteveYin\\MyCode\\datalog-mvp-scaler\\data_local"
            )
            # Flir camera, multiple captures
            mrun: str = "mrun_logs_on_testbed_20MP_202304251740_18"  # front
            # mrun: str = "mrun_logs_on_testbed_20MP_202304251740_5"  # front
            # mrun: str = "mrun_logs_on_testbed_20MP_202304271449_9"  # back
            # Canon EOS 90D, logs and tubes
            # mrun: str = "mrun_logs_on_testbed_32MP_202303291044"
            # mrun: str = "mrun_tubes_on_testbed_32MP_202303271646"
            # iPhone 14PM recessed log close-up
            # mrun: str = "mrun_logs_on_testbed_12MP_202304111609_8"
            pcd_name: str = f"{mrun}_crop.ply"
            vec2sky, vec2sensor = -ut.yaxis, -ut.zaxis

        case "in-house-lidar":
            pcd_folder = "C:\\Users\\SteveYin\\MyCode\\datalog-mvp-scaler"
            mrun_name = "data_local"
            # obj_name = "lidar_data_apr_12_1frame"
            # pcd_name = "lidar_data_apr12_5ftwd_2fttall_1_frame"
            # pcd_name = "pcd-apr25-in-5ftwd-3.4166ftht-1"
            # pcd_name = "pcd-apr25-in-5ftwd-3.4166ftht-2"
            obj_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-1"
            # pcd_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-2"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-1"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-2"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-1"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-2"
            vec2sky, vec2sensor = ut.zaxis, -ut.xaxis

        case "snoqualmie-phtgm":
            # Snoqualmie test site data 5/23/2023
            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"mrun_snoload{which_load}{which_end}_finefeat"
            vec2sky, vec2sensor = -ut.yaxis, -ut.zaxis

        case "snoqualmie-lidar":
            # Snoqualmie test site data 5/23/2023
            folder_name = "C:\\Users\\SteveYin\\MyData"
            mrun_name = "snofield20230523\\voi_pcd"
            obj_name = f"pcd-{which_load}-{which_snr}-{which_end}-{which_step}"
            vec2sky, vec2sensor = -ut.yaxis, -ut.zaxis

        case _:
            sys.exit("case not handled when loading pcd data file...")

    # load the pcd data
    pcd = ut.load_pcd(
        pcd_path=f"{folder_name}\\{mrun_name}\\{obj_name}_crop.ply"
    )
    ut.sck.is_valid_pcd(pcd, "pcd")
    pcd.scale(ptg2real_scale, pcd.get_center())
    pcd.estimate_normals()

    # instantiate a PointcloundHandler instance
    voi = PointcloudHandler(
        pcd_raw=pcd,
        pcd_idx=[-1] * len(pcd.points),
        vec2sky=vec2sky,
        vec2sensor=vec2sensor,
        disp_info=True,
        disp_progress=True,
    )

    # run test cases
    color_idx: list[int]
    match which_test:
        case "pcd_color_pick":
            # test color picking
            cc = matplotlib.colors.rgb_to_hsv(np.array(voi.pcd.colors))
            aa = voi.pcd.colors
            bb = np.tile(
                np.array([128 / 256, 128 / 256, 64 / 256]), (len(aa), 1)
            )
            dd = aa - bb
            color_idx = np.where(dd[:, 2] >= 0)[0].tolist()
            color_pcd = voi.pcd.select_by_index(color_idx)
            color_pcd.paint_uniform_color(ut.CP.BLUE_DARK)
            o3d.visualization.draw(
                [
                    {"name": "pcd", "geometry": voi.pcd},
                    {"name": "color_pts", "geometry": color_pcd},
                ],
                show_ui=True,
                # point_size=5,
            )

        case "pcd_idx_label":
            # test color picking & set pcd color
            cc = matplotlib.colors.rgb_to_hsv(np.array(voi.pcd.colors))
            aa = voi.pcd.colors
            bb = np.tile(
                np.array([128 / 256, 128 / 256, 64 / 256]), (len(aa), 1)
            )
            dd = aa - bb
            color_idx = np.where(dd[:, 2] >= 0)[0].tolist()

            alabel = 1
            voi.set_pcd_idx_label(selected_idx=color_idx, label=alabel)
            idx_chosen = np.where(np.array(voi.pcd_idx) == alabel)[0].tolist()
            voi.set_pcd_color(idx_chosen, ut.CP.CYAN_DARK)
            o3d.visualization.draw(
                [
                    {"name": "pcd", "geometry": voi.pcd},
                ],
                show_ui=True,
                # point_size=5,
            )

        case "pcd_normals_pick":
            # test using normal vec as a condition for log seg
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

            # select points that align with vec2sensor direction
            aa = np.array(voi.pcd.normals)
            bb = np.tile(vec2sensor, (len(aa), 1))
            dd = ut.vector_angle(aa, bb)
            dd_idx = np.where((dd <= np.pi / 6) | (dd >= np.pi / 6 * 5))[
                0
            ].tolist()
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
                grd_hgt_lo=grd_hgt_lo,
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

            """
            # run planar patch detction to segment the log ends
            oboxes = cl.detect_planar_patches(
                normal_variance_threshold_deg=25,  # 25
                coplanarity_deg=60,  # 60
                outlier_ratio=0.25,  # 0.25
                min_plane_edge_length=0,
                min_num_points=10,  # 10
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5),  # 10
            )
            ut.draw_planar_patches(
                oboxes=oboxes,
                o3dobj=cl,
                uvw_scale=0.2,
                uvw_selected=np.array([True, True, True]),
                disp_info=True,
            )
            # get segments from oboxes
            valid_idx = [False] * len(oboxes)
            np_colors = np.array(voi.pcd.colors)
            for jj, this_obox in enumerate(oboxes):
                ctr = this_obox.center
                eu, ev, ew = (
                    this_obox.extent[0],
                    this_obox.extent[1],
                    this_obox.extent[2],
                )
                aspect_ratio = np.max([eu / ev, ev / eu])
                if aspect_ratio < 2 and aspect_ratio > 0.5:
                    valid_idx[jj] = True
                    # flip obox if its ww vector aligns to -vec2sensor
                    obox, _, _ = ut.flip_obox(this_obox, vec2sensor)
                    colors = plt.get_cmap("tab20")(jj)
                    obox.color = list(colors[:3])
                    idx = obox.get_point_indices_within_bounding_box(
                        voi.pcd.points
                    )
                    np_colors[idx, :] = list(colors[:3])
            voi.pcd.colors = o3d.utility.Vector3dVector(np_colors)

            oboxes_valid = np.array(oboxes)[valid_idx].tolist()
            ut.draw_planar_patches(
                oboxes=oboxes_valid,
                o3dobj=voi.pcd,
                uvw_scale=0.2,
                uvw_selected=np.array([True, True, True]),
                disp_info=True,
            )
            o3d.visualization.draw(
                [
                    {"name": "pcd", "geometry": voi.pcd},
                ],
                show_ui=True,
                # point_size=5,
            )
            """

        case "pcd_handler":
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
            # and logend segmentation
            voi.planar_patches(
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
                uvw_scale=0.1,
                uvw_selected=np.array([True, True, True]),
                disp_info=True,
            )
            voi.get_ref_plane(
                pose_ang_lo=np.pi / 6,
                pose_ang_hi=np.pi / 6 * 5,
                ratio_uv_lo=0.33,
                ratio_uv_hi=3.0,
            )

            ct = voi.pcd.get_center()
            mvec2sky = ut.get_arrow_mesh(
                ct, ct + vec2sky, scale=0.6, arrow_color=ut.CP.BLUE_DARK
            )
            mvec2sensor = ut.get_arrow_mesh(
                ct, ct + vec2sensor, scale=0.6, arrow_color=ut.CP.GREEN_DARK
            )
            mvec2 = mvec2sky + mvec2sensor

            _, snr_plane_obox_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=voi.snr_plane_obox,
                obox_color=ut.CP.RED_DARK,
                uvw_scale=0.4,
                uvw_selected=np.array([True, True, True]),
                mu_color=ut.CP.RED,
                mv_color=ut.CP.GREEN,
                mw_color=ut.CP.BLUE,
            )

            _, grd_plane_obox_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=voi.grd_plane_obox,
                obox_color=ut.CP.ORANGE_DARK,
                uvw_scale=0.4,
                uvw_selected=np.array([True, True, True]),
                mu_color=ut.CP.RED,
                mv_color=ut.CP.GREEN,
                mw_color=ut.CP.BLUE,
            )

            _, ref_plane_obox_uvw = ut.get_o3dobj_obox_and_uvw(
                o3dobj=voi.ref_plane_obox,
                obox_color=ut.CP.BLUE_DARK,
                uvw_scale=0.4,
                uvw_selected=np.array([True, True, True]),
                mu_color=ut.CP.RED,
                mv_color=ut.CP.GREEN,
                mw_color=ut.CP.BLUE,
            )

            o3d.visualization.draw(
                [
                    {"name": "pcd", "geometry": voi.pcd},
                    {"name": "snr_arr", "geometry": voi.snr_arr_mesh},
                    {"name": "snr_plane", "geometry": voi.snr_plane_obox},
                    {
                        "name": "snr_plane_uvw",
                        "geometry": snr_plane_obox_uvw,
                    },
                    {"name": "snr_plane_mesh", "geometry": voi.snr_plane_mesh},
                    {"name": "grd_plane", "geometry": voi.grd_plane_obox},
                    {
                        "name": "grd_plane_uvw",
                        "geometry": grd_plane_obox_uvw,
                    },
                    {"name": "grd_plane_mesh", "geometry": voi.grd_plane_mesh},
                    {"name": "ref_plane", "geometry": voi.ref_plane_mesh},
                    {
                        "name": "ref_plane_uvw",
                        "geometry": ref_plane_obox_uvw,
                    },
                    {"name": "vec2sky | vec2sensor", "geometry": mvec2},
                ],
                show_ui=True,
                # point_size=5,
            )
