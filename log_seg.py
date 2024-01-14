# ----------------------------------------------------------------------------
# - Pointcloud Data Processing for log-end segmentation & measurement        -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
import open3d as o3d
import numpy as np
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import utility as ut
from pointcloudhandler import PointcloudHandler


if __name__ == "__main__":
    phtgm_or_lidar = True
    disp_progress = True

    # choose clustering options
    opt_clustering = [
        "KMEANS",
        "DBSCAN",
        "HDBSCAN",
        "ECULIDEAN_DBSCAN",
        "PLANE_RANSAC_DBSCAN",
        "PLANE_RANSAC",
        "PLANE_PATCH_PHTGM",
        "PLANE_PATCH_LIDAR"
    ]
    use_clustering = opt_clustering[6]

    if use_clustering == opt_clustering[6]:
        phtgm_or_lidar = True

    if use_clustering == opt_clustering[7]:
        phtgm_or_lidar = False

    if phtgm_or_lidar:
        mrun = "mrun_logs_on_testbed_20MP_202304251740_5"
        # mrun = "mrun_logs_on_testbed_32MP_202303291044"
        # mrun = "mrun_logs_on_testbed_20MP_202304251740"
        # mrun = "mrun_logs_on_testbed_20MP_202304271449_9"

        mesh_folder = "C:\\Users\\SteveYin\\MyCode\\datalog-photogrammetry"
        mesh_name = "texturedMesh.obj"
        pcd_original, mesh_original = ut.load_pcd_from_mesh(
            mesh_path=f"{mesh_folder}\\{mrun}\\{mesh_name}"
        )

        pcd_folder = (
            "C:\\Users\\SteveYin\\MyCode\\datalog-pcd-processing\\data"
        )
        pcd_name = f"{mrun}_crop.ply"

        # dbscan parameters for photogrammetry cases (32MP cam)
        eps_value = 0.005       # dbscan_epsilon value for dbscan
        nth_neighbors = 50      # nth_neighbors value for calc epsilon
        min_cluster_pts = 50    # min_cluster_points value for dbscan

    else:
        pcd_folder = (
            "C:\\Users\\SteveYin\\MyCode\\datalog-pcd-processing\\data"
        )
        pcd_name = "lidar_data_apr_12_1frame_crop.ply"

        # dbscan parameters for lidar cases (livox)
        eps_value = 0.01        # dbscan_epsilon value for dbscan
        nth_neighbors = 30      # nth_neighbors value for calc epsilon
        min_cluster_pts = 30    # min_cluster_points value for dbscan

    # ransac parameters
    npt_ransac = 3          # number of pts for ransac, 3 for a plane
    dist_value = 0.01       # distance_thershold value for ransac
    max_num_planes = 30     # max_num_planes value for ransac
    num_ransac_iters = 500  # num_iterations for ransac

    # load pced into Open3D, then instantiate a PointcloundHandler instance
    pcd = ut.load_pcd(pcd_path=f"{pcd_folder}\\{pcd_name}")
    voi = PointcloudHandler(pcd)
    voi.disp_info = True
    voi.disp_progress = disp_progress
    o3d.visualization.draw([voi.pcd, ut.get_xyz_axes(frame_size=0.5)])

    # a variety of segmentation options
    match use_clustering:
        # KMEANS clustering thru sklearn
        case "KMEANS":
            model = KMeans(n_clusters=20, n_init="auto")
            model.fit(StandardScaler().fit_transform(np.array(voi.pcd.points)))
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"sklearn_kmeans detected {len(set(model.labels_))} "
                    f"clusters"
                )
            ut.display_labeled_clusters(voi.pcd, cluster_labels=model.labels_)

        # DBSCAN clustering thru sklearn:
        case "DBSCAN":
            model = DBSCAN(
                eps=voi.get_dbscan_epsilon(),
                min_samples=50,
                n_jobs=-1
            )
            model.fit(StandardScaler().fit_transform(np.array(voi.pcd.points)))
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"sklearn_dbscan detected {len(set(model.labels_))} "
                    f"clusters"
                )
            ut.display_labeled_clusters(voi.pcd, cluster_labels=model.labels_)

        # HDBSCAN clustering thru hdbscan:
        case "HDBSCAN":
            model = hdbscan.HDBSCAN(
                algorithm="best",
                min_cluster_size=10,
                cluster_selection_epsilon=0.0,
                metric="euclidean",
            ).fit(StandardScaler().fit_transform(np.array(voi.pcd.points)))
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"hdbscan detected {len(set(model.labels_))} "
                    f"clusters"
                )
            ut.display_labeled_clusters(voi.pcd, cluster_labels=model.labels_)

        # ECULIDEAN_DBSCAN thru O3D
        case "ECULIDEAN_DBSCAN":
            voi.euclidean_dbscan(
                epsilon=eps_value,
                min_cluster_points=min_cluster_pts
            )
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"eculidean_dbscan detected "
                    f"{len(set(voi.labels_euclidean_dbscan))} clusters"
                )
            ut.display_labeled_clusters(
                pcd=voi.pcd,
                cluster_labels=voi.labels_euclidean_dbscan
            )
            # filter the detected clusters to appropriate sized clusters
            max_label = voi.labels_eculidean_dbscan.max()
            segments = []
            for jj in range(max_label):
                this_pcd = voi.pcd.select_by_index(
                    list(np.where(voi.labels_eculidean_dbscan == jj)[0])
                )
                if len(this_pcd.points) <= 1000:
                    continue
                obox = this_pcd.get_oriented_bounding_box()
                extent = obox.extent
                rotmat = obox.R
                ex, ey, ez = extent[0], extent[1], extent[2]
                ang = np.arccos(
                    np.dot(np.asarray(rotmat[:, 2]), np.asarray([0, 0, -1]))
                ) / np.pi * 180
                if (
                    ang <= 45
                    and (ex**2 + ey**2) >= 0.0625
                    and ex / ey > 0.5 and ex / ey < 1.5
                ):
                    segments.append(this_pcd)
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"size & pose filtering selected {len(segments)} clusters"
                )
            pcd_new, label_new = ut.assemble_pcd_clusters(pcd_clusters=segments)
            ut.get_o3dobj_info(
                o3dobj=pcd_new, o3dobj_name="new pcd", disp_info=True
            )
            ut.display_labeled_clusters(pcd=pcd_new, cluster_labels=label_new)

        # PLANE_RANSAC_DBSCAN thru O3D
        case "PLANE_RANSAC_DBSCAN":
            voi.plane_ransac_dbscan(
                epsilon=eps_value,
                min_cluster_points=min_cluster_pts,
                npt_ransac=npt_ransac,
                d_threshold=dist_value,
                max_num_planes=max_num_planes,
                num_ransac_iters=num_ransac_iters
            )
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"plane_ransac_dbscan detected "
                    f"{len(voi.ransac_dbscan_segments)} clusters"
                )
            o3d.visualization.draw([
                voi.ransac_dbscan_segments, ut.get_xyz_axes(frame_size=0.5)
            ])

        # PLANE_RANSAC thru O3D
        case "PLANE_RANSAC":
            voi.plane_ransac(
                npt_ransac=npt_ransac,
                d_threshold=dist_value,
                num_ransac_iters=num_ransac_iters,
                max_num_planes=max_num_planes
            )
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"plane_ransac detected "
                    f"{len(voi.ransac_plane_segments)} clusters"
                )
            o3d.visualization.draw(
                [voi.ransac_plane_segments, ut.get_xyz_axes(frame_size=0.5)]
            )

        # PLANE_PATCH thru O3D - dealing with photogrammetry-generated mesh/pcd
        case "PLANE_PATCH_PHTGM":
            # pcd_ds = voi.down_sample_uniform(every_k_point=10)
            pcd_clean, ind = voi.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=0.75
            )
            obox_voi = pcd_clean.get_oriented_bounding_box()
            uu = np.asarray(obox_voi.R[:, 0])
            vv = np.asarray(obox_voi.R[:, 1])
            ww = np.asarray(obox_voi.R[:, 2])
            mu = ut.get_arrow(
                vector=uu, scale=0.02, arrow_color=np.asarray([1.0, 0.0, 0.0])
            )
            mv = ut.get_arrow(
                vector=vv, scale=0.02, arrow_color=np.asarray([0.0, 1.0, 0.0])
            )
            mw = ut.get_arrow(
                vector=ww, scale=0.02, arrow_color=np.asarray([0.0, 0.0, 1.0])
            )

            oboxes = pcd_clean.detect_planar_patches(
                normal_variance_threshold_deg=20,   # 60
                coplanarity_deg=75,     # 75
                outlier_ratio=0.15,  # 0.75
                min_plane_edge_length=0,  # 0
                min_num_points=0,    # 0
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)    # 30
                # normal_variance_threshold_deg=60,   # 60
                # coplanarity_deg=50,     # 75
                # outlier_ratio=0.25,  # 0.75
                # min_plane_edge_length=0,  # 0
                # min_num_points=800,    # 0
                # search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10)  # 30
            )
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"     voi.pcd.oriented_bounding_box info: \n"
                    f"     uu=[{uu[0]:.2e}, {uu[1]:.2e}, {uu[2]:.2e}],\n"
                    f"     vv=[{vv[0]:.2e}, {vv[1]:.2e}, {vv[2]:.2e}],\n"
                    f"     uu=[{uu[0]:.2e}, {uu[1]:.2e}, {uu[2]:.2e}],\n"
                    f"     Planar Patch Detected {len(oboxes)} patches\n"
                )
                o3d.visualization.draw(
                    oboxes
                    + [pcd_clean, ut.get_xyz_axes(frame_size=0.5), mu, mv, mw]
                )

            selected_index = ut.filter_oriented_bounding_boxes(
                obox_candidates=oboxes,
                ref_vec=-ut.zaxis,
                ref_ang_lo=70.0,
                ref_ang_hi=110.0,
                ref_diag_lo=0.15,
                ref_diag_hi=0.9,
                ref_ratio_xy_lo=0.33,
                ref_ratio_xy_hi=3.0,
                disp_info=True
            )
            oboxes_selected = np.asarray(oboxes)[selected_index].tolist()            
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"Further selected {len(oboxes_selected)} patches"
                )
            ut.save_planar_patches(oboxes=oboxes[selected_index is True])
            ut.draw_planar_patches(oboxes=oboxes_selected, base_obj=voi.pcd)

        # PLANE_PATCH thru O3D - dealing with lidar pcd
        case "PLANE_PATCH_LIDAR":
            # using all defaults
            oboxes = voi.pcd.detect_planar_patches(
                normal_variance_threshold_deg=60,   # 60
                coplanarity_deg=75,     # 75
                outlier_ratio=0.75,  # 0.75
                min_plane_edge_length=0,  # 0
                min_num_points=0,    # 0
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)     # 30
            )
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"Planar Patch Detected {len(oboxes)} patches"
                )
            o3d.visualization.draw(oboxes)

            selected_index = ut.filter_oriented_bounding_boxes(
                obox_candidates=oboxes,
                ref_vec=-ut.xaxis,
                ref_ang_lo=45.0,
                ref_ang_hi=135.0,
                ref_diag_lo=0.15,
                ref_diag_hi=0.9,
                ref_ratio_xy_lo=0.5,
                ref_ratio_xy_hi=2.0,
                disp_info=True
            )
            oboxes_selected = np.asarray(oboxes)[selected_index].tolist()            
            if disp_progress:
                print(
                    f"[INFO: {ut.currentFuncName()}]: "
                    f"Further selected {len(oboxes_selected)} patches"
                )
            ut.save_planar_patches(oboxes=oboxes_selected)
            ut.draw_planar_patches(oboxes=oboxes_selected, base_obj=voi.pcd)

        # Unknown handling
        case _:
            print(
                f"[INFO: {ut.currentFuncName()}]: "
                f"use_clustering definition not supported in "
                f"opt_clustering({opt_clustering})")

    try:
        userInput = input(f"[INFO: {ut.currentFuncName()}]: Press q to exit")
        if userInput == "q":
            exit()
    except SyntaxError:
        pass
