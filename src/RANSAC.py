import numpy as np
from scipy.spatial.distance import pdist, squareform  # 可选，用于优化距离计算

filename_strs = [
    "box_paper",
    "box_paper_withangle",
    "iron_rect_black_withangle",
    "iron_rect1_black",
]

idx = 3
model_name = "x"
image_sz = 1024

threshold = 0.015  # 内点阈值 (根据噪声水平调整)
max_iterations = 20000  # 最大迭代次数
min_inliers_ratio = 0.6  # 最小内点比例


temp_name = filename_strs[idx - 1] + f"_fastsam_{model_name}_{image_sz}"


def estimate_plane_ransac(
    points, threshold=0.01, max_iterations=1000, min_inliers=None
):
    """
    使用 RANSAC 算法估计三维点云中的最佳平面方程。

    平面方程形式: Ax + By + Cz + D = 0，其中 (A, B, C) 是平面法向量。

    参数:
    points (np.ndarray): 输入点云数据，形状为 (N, 3)。
    threshold (float): 内点与估计平面的最大距离阈值。
    max_iterations (int): RANSAC 算法的最大迭代次数。
    min_inliers (int, optional): 识别为最佳平面的最小内点数量。
                                 如果为 None，则默认为点云总数的 20%。

    返回:
    tuple: (best_plane_coeffs, best_inliers_mask)
           best_plane_coeffs (np.ndarray): 最佳平面的系数 [A, B, C, D]。
           best_inliers_mask (np.ndarray): 布尔数组，指示哪些点是最佳平面的内点。
                                          如果未找到合适平面，则返回 (None, None)。
    """
    num_points = points.shape[0]
    if num_points < 3:
        raise ValueError("点云中的点数必须至少为 3。")

    if min_inliers is None:
        min_inliers = int(0.2 * num_points)  # 默认至少20%的点是内点

    best_inliers_count = 0
    best_plane_coeffs = None
    best_inliers_mask = np.zeros(num_points, dtype=bool)

    for i in range(max_iterations):
        # 1. 随机选择 3 个点
        # 使用 np.random.choice 更高效地选择不重复的索引
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sample_points = points[sample_indices, :]

        p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]

        # 检查选取的三个点是否共线
        v1 = p2 - p1
        v2 = p3 - p1
        # 计算叉积的模长，如果接近0则共线
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue  # 如果共线，则跳过本次迭代

        # 2. 从这 3 个点估计平面方程 Ax + By + Cz + D = 0
        # 法向量 N = (p2 - p1) x (p3 - p1)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # 归一化法向量

        # D = - (A*x0 + B*y0 + C*z0)，使用其中一个点来计算 D
        D = -np.dot(normal, p1)  # normal 是 [A, B, C]

        current_plane_coeffs = np.array([normal[0], normal[1], normal[2], D])

        # 3. 计算所有点到当前估计平面的距离
        # Ax + By + Cz + D
        numerator = np.abs(np.dot(points, normal) + D)
        # 分母是法向量的模长，但我们已经归一化了法向量，所以模长是 1
        distances = numerator / np.linalg.norm(
            normal
        )  # 这里 np.linalg.norm(normal) 已经是 1

        # 4. 统计内点（距离小于阈值的点）
        inliers_mask = distances < threshold
        current_inliers_count = np.sum(inliers_mask)

        # 5. 更新最佳模型
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_plane_coeffs = current_plane_coeffs
            best_inliers_mask = inliers_mask

        # 提前终止条件：如果内点数量足够多，可以停止迭代
        if best_inliers_count >= min_inliers:
            print(f"RANSAC 提前终止：找到足够多的内点 ({best_inliers_count})。")
            break

    if best_plane_coeffs is not None and best_inliers_count >= min_inliers:
        print(f"RANSAC 完成。最佳平面包含 {best_inliers_count} 个内点。")
        # （可选）可以对最佳内点重新拟合平面，以获得更精确的模型
        # best_inlier_points = points[best_inliers_mask]
        # best_plane_coeffs = refine_plane_from_points(best_inlier_points)
        return best_plane_coeffs, best_inliers_mask
    else:
        print(
            f"RANSAC 未能找到满足条件的平面。最佳内点数量为 {best_inliers_count} (要求至少 {min_inliers})。"
        )
        return best_plane_coeffs, best_inliers_mask


def refine_plane_from_points(points):
    """
    从一组点中通过奇异值分解（SVD）精确拟合平面。
    这通常在 RANSAC 找到内点后用于优化平面模型。
    """
    if points.shape[0] < 3:
        raise ValueError("拟合平面至少需要 3 个点。")

    # 计算点的质心
    centroid = np.mean(points, axis=0)
    # 将点云中心化
    centered_points = points - centroid

    # 使用 SVD 找到最小二乘意义下的最佳拟合平面
    # 平面的法向量是协方差矩阵的最小奇异值对应的右奇异向量
    U, s, V = np.linalg.svd(centered_points)
    # 最小奇异值对应的向量是 V 的最后一列 (或 V.T 的最后一行)
    normal = V[-1, :]

    # 计算 D
    D = -np.dot(normal, centroid)

    return np.array([normal[0], normal[1], normal[2], D])


def get_plane_info(point_cloud, threshold, min_inliers_ratio, max_iterations=5e3):
    estimated_coeffs, inliers_mask = estimate_plane_ransac(
        point_cloud,
        threshold,  # 内点阈值 (根据噪声水平调整)
        int(max_iterations),  # 最大迭代次数
        min_inliers=min_inliers_ratio
        * int(len(point_cloud)),  # 最小内点数量 (可根据预期调整)
    )
    if estimated_coeffs is not None:
        print("\n--- RANSAC 估计结果 ---")
        A, B, C, D = estimated_coeffs
        print(f"估计平面方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
        print(f"内点数量: {np.sum(inliers_mask)}")

        # 验证结果：检查内点到估计平面的距离是否都在阈值内
        inlier_points = point_cloud[inliers_mask]
        normal_est = estimated_coeffs[:3]
        D_est = estimated_coeffs[3]
        distances_to_estimated_plane = np.abs(
            np.dot(inlier_points, normal_est) + D_est
        ) / np.linalg.norm(normal_est)
        print(f"最大内点距离: {np.max(distances_to_estimated_plane):.4f}")
        print(
            f"所有内点距离是否小于阈值: {np.all(distances_to_estimated_plane < threshold)}"
        )
        estimated_normal = estimated_coeffs[:3]
        print(
            f"估计平面法向量: [{estimated_normal[0]:.4f}, {estimated_normal[1]:.4f}, {estimated_normal[2]:.4f}]"
        )
        print(f"法向量模长（应接近1）: {np.linalg.norm(estimated_normal):.4f}")
        return [float(x) for x in [A, B, C, D]]
    else:
        return None


# --- 示例用法 ---
if __name__ == "__main__":
    point_cloud = np.load(f"./output/{temp_name}.npz")
    point_cloud = point_cloud["points"]
    print(f"num fo points {len(point_cloud)}")
    # print(f"点云总数: {point_cloud.shape[0]}")
    # print(f"真实平面方程: {A_true}x + {B_true}y + {C_true}z + {D_true} = 0")

    # 使用真实的输入文件
    # 2. 运行 RANSAC
    # 调整阈值和迭代次数以适应你的数据
    estimated_coeffs, inliers_mask = estimate_plane_ransac(
        point_cloud,
        threshold,  # 内点阈值 (根据噪声水平调整)
        max_iterations,  # 最大迭代次数
        min_inliers=min_inliers_ratio
        * int(len(point_cloud)),  # 最小内点数量 (可根据预期调整)
    )

    if estimated_coeffs is not None:
        print("\n--- RANSAC 估计结果 ---")
        A, B, C, D = estimated_coeffs
        print(f"估计平面方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
        print(f"内点数量: {np.sum(inliers_mask)}")

        # 验证结果：检查内点到估计平面的距离是否都在阈值内
        inlier_points = point_cloud[inliers_mask]
        normal_est = estimated_coeffs[:3]
        D_est = estimated_coeffs[3]
        distances_to_estimated_plane = np.abs(
            np.dot(inlier_points, normal_est) + D_est
        ) / np.linalg.norm(normal_est)
        print(f"最大内点距离: {np.max(distances_to_estimated_plane):.4f}")
        print(f"所有内点距离是否小于阈值: {np.all(distances_to_estimated_plane < 0.1)}")

        # 可视化 (需要 matplotlib)
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # 绘制所有点
            ax.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1],
                point_cloud[:, 2],
                color="gray",
                s=10,
                label="all points",
                alpha=0.5,
            )

            # 绘制内点
            ax.scatter(
                point_cloud[inliers_mask, 0],
                point_cloud[inliers_mask, 1],
                point_cloud[inliers_mask, 2],
                color="blue",
                s=20,
                label="inner points",
            )

            # 绘制估计的平面 (需要一些辅助点)
            if C != 0:  # 避免除以零
                xx, yy = np.meshgrid(
                    np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10),
                    np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10),
                )
                zz = (-A * xx - B * yy - D) / C
                ax.plot_surface(
                    xx, yy, zz, alpha=0.2, color="green", label="Estimated Surface"
                )
            elif B != 0:  # 如果C为0，可能是垂直于XY平面的平面
                xx, zz = np.meshgrid(
                    np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10),
                    np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 10),
                )
                yy = (-A * xx - C * zz - D) / B
                ax.plot_surface(
                    xx, yy, zz, alpha=0.2, color="green", label="Estimated Surface"
                )
            elif A != 0:  # 如果B, C为0，可能是垂直于YZ平面的平面
                yy, zz = np.meshgrid(
                    np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10),
                    np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 10),
                )
                xx = (-B * yy - C * zz - D) / A
                ax.plot_surface(
                    xx, yy, zz, alpha=0.2, color="green", label="Estimated Surface"
                )
            else:
                print("法向量为零，无法绘制平面。")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("RANSAC Estimated Surface")
            plt.legend()
            plt.show()

        except ImportError:
            print("未安装 Matplotlib，无法进行可视化。请安装：pip install matplotlib")

        # **在这里单独输出法向量**
        estimated_normal = estimated_coeffs[:3]
        print(
            f"估计平面法向量: [{estimated_normal[0]:.4f}, {estimated_normal[1]:.4f}, {estimated_normal[2]:.4f}]"
        )
        print(f"法向量模长（应接近1）: {np.linalg.norm(estimated_normal):.4f}")
        # **输出法向量结束**
    else:
        print("未找到有效平面。")
