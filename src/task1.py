import open3d as o3d
import numpy as np
import os
import cv2
import time
import gsound

storage_path = "data/"
isRunning = True
lis_coords = [[0, 0, 0.5], [0, 0, 0.6], [0, 0, 0.54], [0, 0, 0.4]]
filename_strs = [
    "box_paper",
    "box_paper_withangle",
    "iron_rect_black_withangle",
    "iron_rect1_black",
]


def key_callback(vis, action, mods):
    global isRunning
    if action == 0:
        isRunning = False


idx = 4
timetick0 = time.time()
# 复现点云
idx -= 1
isRunning = True
pt_data = np.load(f"../data/{filename_strs[idx]}.npz")
points = pt_data["points"] / 1000  # 转换格式
colors = pt_data["colors"]
points[:, -1] += 0.02  # 误差修正

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

Height = 40 * 9
Width = 40 * 16
channels = 3
assert Height * Width == points.shape[0]

# 查看图片
rescale_colors = 255 * colors
rescale_colors = rescale_colors.astype(np.uint8).reshape((Height, Width, channels))

print("H: ", rescale_colors.shape[0], " W: ", rescale_colors.shape[1])

pcd_masked = o3d.geometry.PointCloud()
np_points = np.array(pcd.points)
np_colors = np.array(pcd.colors)
mask1 = np_points[:, 2] <= 1.5
mask2 = np_points[:, 2] > 0.5
mask = np.logical_and(mask1, mask2)

pcd_masked.points = o3d.utility.Vector3dVector(np_points[mask])
pcd_masked.colors = o3d.utility.Vector3dVector(np_colors[mask])
timetick1 = time.time()
result = gsound.mesh_and_trace(
    pcd_masked,
    lis_coord=lis_coords[idx],
    output_path="../obj1/",
    file_name=f"reconstructed_mesh_{idx+1}",
)
timetick2 = time.time()
print("sample rate: ", result[1])

print(
    f"Time statics: preprocess {1000*(timetick1-timetick0)} ms. mesh and trace {1000*(timetick2-timetick1)} ms."
)
