# ----------------------------------------------------------------------------
# - Pointcloud Data loading, croping and visualization                       -
# ----------------------------------------------------------------------------
# - FOR DATALOG MVP PHASE 1 @ FRESH CONSULTING                               -
# ----------------------------------------------------------------------------
import sys
import open3d as o3d
import utility as ut
import copy


if __name__ == "__main__":
    data_type: list[str] = [
        "in-house-phtgm-mesh",
        "in-house-lidar",
        "snoqualmie-phtgm",
        "snoqualmie-lidar",
    ]
    which_data: str = data_type[3]
    load_num: int = 3  # 1,2,3,4 for snoqualmie onsite data
    # which_end: str = "bck"  # 'frt' or 'bck' for snoqualmie onsite phtgm data
    which_end: str = "back"  # 'frt' or 'bck' for snoqualmie onsite phtgm data
    which_snr: int = 102
    which_row: int = 1
    mesh_fraction: float = 1.0
    show_intermediate: bool = False
    folder_name: str = ""
    mrun_name: str = ""
    obj_name: str = ""

    match which_data:
        case "in-house-phtgm-mesh":
            # Flir camera, multiple captures
            folder_name = "C:\\Users\\SteveYin\\MyCode\\datalog-photogrammetry"
            obj_name = "texturedMesh.obj"
            mrun_name = "mrun_logs_on_testbed_20MP_202304251740_18"
            # mrun_name = "mrun_logs_on_testbed_20MP_202304251740_5"  # back
            # mrun_name = "mrun_logs_on_testbed_20MP_202304271449_9"  # front
            # Canon EOS 90D, logs and tubes
            # mrun_name = "mrun_logs_on_testbed_32MP_202303291044"
            # mrun_name = "mrun_tubes_on_testbed_32MP_202303271646"
            # iPhone 14PM recessed log close-up
            # mrun_name = "mrun_logs_on_testbed_12MP_202304111609_8"

            # load mesh into Open3D, then extract pcd from the mesh
            pcd_original, mesh_original = ut.load_pcd_from_mesh(
                mesh_path=f"{folder_name}\\{mrun_name}\\{obj_name}"
            )
            if show_intermediate:
                o3d.visualization.draw(
                    [mesh_original, ut.get_xyz_axes(frame_size=0.5)]
                )
            # simplify mesh if desired by user
            if mesh_fraction < 1.0 and mesh_fraction >= 0.1:
                mesh_smp = mesh_original.simplify_quadric_decimation(
                    target_number_of_triangles=(int)(
                        len(mesh_original.triangles) * mesh_fraction
                    )
                )
                ctr, bmn, bmx = ut.get_o3dobj_info(o3dobj=mesh_smp)
                pcd = o3d.geometry.PointCloud()
                pcd.points = mesh_smp.vertices
                pcd.colors = mesh_smp.vertex_colors
                pcd.normals = mesh_smp.vertex_normals
            else:
                pcd = copy.deepcopy(pcd_original)
                pcd.estimate_normals()
            vec2sky, vec2sensor = -ut.yaxis, -ut.zaxis

        case "snoqualmie-phtgm":
            # Snoqualmie test site data 5/23/2023
            folder_name = (
                "C:\\Users\\SteveYin\\MyData\\snofield20230523\\sfm_densepcd"
            )
            obj_name = "sfm.ply"
            mrun_name = f"mrun_snoload{load_num}{which_end}_finefeat"

            pcd = ut.load_pcd(
                pcd_path=f"{folder_name}\\{mrun_name}\\{obj_name}"
            )
            pcd.estimate_normals()
            vec2sky, vec2sensor = -ut.yaxis, -ut.zaxis

        case "snoqualmie-lidar":
            # Snoqualmie test site data 5/23/2023
            folder_name = "C:\\Users\\SteveYin\\MyData\\snofield20230523"
            obj_name = f"pcd-{load_num}-{which_snr}-{which_end}-{which_row}.pcd"
            # mrun_name = f"mrun_snoload{load_num}{which_end}_finefeat"

            pcd = ut.load_pcd(pcd_path=f"{folder_name}\\{obj_name}")
            pcd.estimate_normals()
            vec2sky, vec2sensor = -ut.yaxis, ut.xaxis

        case "in-house-lidar":
            folder_name = "C:\\Users\\SteveYin\\MyCode\\datalog-lidar"
            mrun_name = "data"
            pcd_name = "lidar_data_apr_12_1frame.pcd"
            pcd_name = "lidar_data_apr12_5ftwd_2fttall_1_frame_crop.ply"
            # mrun_name = "LiDAR-Indoor-5ftWD"
            # pcd_name = "pcd-apr25-in-5ftwd-3.4166ftht-1.pcd"
            # pcd_name = "pcd-apr25-in-5ftwd-3.4166ftht-2.pcd"
            # pcd_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-1.pcd"
            # pcd_name = "pcd-apr27-in-5ftwd-3.4166ftht-back-2.pcd"
            # mrun_name = "LiDAR-Outdoor-5ftWD"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-1.pcd"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-front-2.pcd"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-1.pcd"
            # pcd_name = "pcd-apr27-out-5ftwd-3.4166ftht-back-2.pcd"

            # load pcd into Open3D
            pcd = ut.load_pcd(
                pcd_path=f"{folder_name}\\{mrun_name}\\{pcd_name}"
            )
            pcd.estimate_normals()
            vec2sky, vec2sensor = ut.zaxis, -ut.xaxis

        case _:
            sys.exit("case not handled")

    # crop pcd voi for processing in log_scaling.py and log_scaling2.py
    ut.crop_pcd_voi(pcd)
