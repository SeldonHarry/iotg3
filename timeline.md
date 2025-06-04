# 记录研究过程的计划
## 4.3
+ 从stereo信息恢复到spatial信息的例程：https://github.com/luxonis/depthai-experiments/tree/master/gen2-calc-spatials-on-host
+ task2的光线追踪：助教建议使用BVH方法来实现。有三个著名方案（都是基于Cpp的）：[embree](https://github.com/embree/embree)，[Optix](https://developer.nvidia.com/optix)，[PBRT](https://github.com/mmp/pbrt-v3)。
+ 也可以进一步考虑 https://github.com/brandonpelfrey/Fast-BVH
+ 第一步的目标其实是从点云数据中得到一个合理的反射面。也就是说，对于含有不少噪声和缺陷的数据要进行处理，得到期望的平面。论文中提到使用RANSAC算法来做处理。这个算法可以用python实现。因此，task1具有使用python实现的一致性。

## 4.4
+ TA建议：不用平面分割，拿到point cloud，转为mesh，然后直接用BVH这样的传统射线追踪跑一跑试试
+ 光线追踪和BVH学习：https://www.bilibili.com/video/av90798049/

## 4.6
+ 设法把深度信息提取的方式找到并储存成数组文件（.cvs or .npz）


### PPT make
+ 展示pygsound接口理解过程: 如何设定表面性质参数
+ 评价不同情况下的点云构建效果（纹理、光照）=》因此选择的反射面上需要贴纸
+ 调研mesh建立方法，选择poisson方法建立mesh
+ 调研FastSAM论文，
    ctx.diffuse_count = 2000000
    ctx.specular_count = 20000 是pygsound中的关键参数

    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1e-3, max_nn=30)
    )  # radius 为平均间距的2-5倍  这也是个关键参数