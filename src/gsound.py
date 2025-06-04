import open3d as o3d
import numpy as np
import os
import cv2
import copy
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time

# 使用pygsound
import pygsound as ps


def pygsound_plot(
    IR: np.ndarray,
    sample_rate: int,
    save_plot=False,
    output_path="./output/",
    file_name="temp",
):
    time_tick = np.arange(len(IR)) / sample_rate
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))
    axs.plot(time_tick, IR)
    axs.set_xlim(0, 0.02)
    axs.set_xlabel("time (s)")
    axs.set_ylim(0, 1)
    axs.set_ylabel("Amplitude")
    fig.tight_layout()
    plt.grid(True)
    if save_plot:
        plt.savefig(output_path + file_name + ".png")
    plt.show()

    return None


def mesh_and_trace(
    pcd_masked,
    lis_coord,
    save_obj=True,
    output_path="./output/",
    file_name="temp",
    view_in_process=False,
    absorption_coefficients=0.5,
    scattering_coefficients=0.0,
):
    """from a masked pointscloud file generate sound ir, src is at [0,0,0]

    Args:
        pcd_masked (np.ndarray): points cloud data
        lis_coord (list): 3D coord
        save_obj (bool, optional): _description_. Defaults to False.
        output_path (str, optional): _description_. Defaults to "./output/".
        file_name (str, optional): _description_. Defaults to "temp".
        view_in_process (bool, optional): _description_. Defaults to False.
        absorption_coefficients (float, optional): _description_. Defaults to 0.5.
        scattering_coefficients (float, optional): _description_. Defaults to 0.0.

    Returns:
        list: list contains single channel ir and sample rate
    """
    start_time = time.time()
    pcd_masked.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1e-3, max_nn=30)
    )  # radius 为平均间距的2-5倍
    # 可以使用 pcd_tets.orient_normals_consistent_tangent_plane(k=10) 来尝试统一法线方向
    pcd_masked.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )

    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_masked, depth=8, width=0, scale=1.1, linear_fit=False
    )
    vertices_to_remove = densities < np.quantile(
        densities, 0.05
    )  # 移除密度最低的5%的顶点
    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)

    if view_in_process:
        o3d.visualization.draw_geometries(
            [mesh_poisson],
            window_name="Poisson Reconstruction",
            point_show_normal=False,  # 可以设为True查看法线
            mesh_show_wireframe=True,
            mesh_show_back_face=True,
        )

    output_obj_file = output_path + file_name + ".obj"
    # 保存
    if save_obj:
        output_obj_file = output_path + file_name + ".obj"
        o3d.io.write_triangle_mesh(
            output_obj_file,
            mesh_poisson,
            write_vertex_normals=True,
            write_vertex_colors=False,
        )
        print("Successfully write possion obj file")
    mid_time = time.time()
    ctx = ps.Context()
    ctx.diffuse_count = 2000000
    ctx.specular_count = 20000
    ctx.channel_type = ps.ChannelLayoutType.stereo

    meshA = ps.loadobj(
        output_obj_file, absorption_coefficients, scattering_coefficients
    )
    scene = ps.Scene()
    scene.setMesh(meshA)
    src_coord = [0, 0, 0]
    src = ps.Source(src_coord)
    src.power = 1.0
    src.radius = 0.05
    lis = ps.Listener(lis_coord)
    lis.radius = 0.05

    res = scene.computeIR(
        [src], [lis], ctx
    )  # you may pass lists of sources and listeners to get N_src x N_lis IRs
    sample_rate = int(res["rate"])
    rir_gs = res["samples"][0][0][0]
    sorted_arg = np.argsort(np.abs(rir_gs))[::-1]
    print(f"Amplitutde pulse 15 strongest idx: {sorted_arg[:15]}")
    end_time = time.time()
    print(
        f"Execute time in function mesh and trace: {1000*(end_time-start_time):.0f} ms. Trace using {1000*(end_time-mid_time):.0f} ms."
    )
    return [rir_gs, sample_rate]
