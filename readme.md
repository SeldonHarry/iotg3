#
使用oak相机

## requirements
Multipath channel estimation plays a crucial role not only in mitigating interference in wireless communication but also in advancing sensing technologies. Traditional channel estimation methods rely on collaborative signal exchanges between transmitters and receivers, limiting their applicability in scenarios where pre-established communication links are unavailable. Recent developments in computer vision, however, offer a promising alternative by enabling real-time multipath estimation with unilateral participation. For example, by using a stereo camera to reconstruct 3D spatial maps between transceivers, ray-tracing simulations can be performed to infer wireless signal propagation paths. This vision-based approach allows for multipath channel prediction from the transmitter or any arbitrary viewpoint, greatly enhancing deployment flexibility. In this project, we aim to develop a vision-based multipath estimation pipeline and evaluate its performance on embedded devices in practical environments. While the proposed method is not limited to specific signal types, we prioritize acoustic waves due to their slower propagation speed, which enhances the detectability of multipath delays compared to electromagnetic signals.

多径信道估计在无线通信中不仅对减轻干扰起着重要作用，而且在传感技术中也起着至关重要的推动作用。传统的信道估计方法依赖于发送器和接收器之间的协作信号交换，这在没有预先建立通信链路的情况下限制了其应用。然而，计算机视觉的最新进展为此提供了一种有前景的替代方案，能够实现单方面参与的实时多径估计。例如，使用立体相机重建发送器和接收器之间的3D空间地图，可以通过射线追踪仿真推断无线信号传播路径。这种基于视觉的方法允许在发送端或任意视角下进行多径信道预测，从而显著提升了部署的灵活性。在本项目中，我们旨在开发一个基于视觉的多径估计管道，并在实际环境中评估其在嵌入式设备上的性能。虽然该方法不限于信号类型，但我们优先考虑声波，因为声波的传播速度较慢，相较于电磁信号，它能放大多径延迟的可辨别性。

1. 3D reconstruciton with binocular camera.
2. Estimate the multipath channel based on the reconstructed 3D model with ray tracing.
3. Leverage AI model to accelerate the pipeline.
4. Implement the pipeline on Raspberry Pi and optimize for the delay.
## 安装方法
基础环境 + python 包
```shell 
sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash
python3 -m pip install depthai # 本机在MCE环境中
```
然后测试安装是否成功：
```shell
git clone https://github.com/luxonis/depthai-python.git
cd depthai-python/examples
python3 install_requirements.py # 要挂代理
python3 ColorCamera/rgb_preview.py # 测试相机
```
用以上命令安装例程实例代码

一些其他依赖包包括open3d 0.18.0
```shell
pip install pygsound 
```
## task1
思路：先得到高清的、无处理的深度数据，并且储存起来
程序具体目标：先打开rgb图像层，然后在按下确认后开始使用双目视觉部分记录图像数据，结束记录后进行深度信息处理.

由于stereo处理部分计算量占主导，参看https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth/#StereoDepth-Stereo%20depth%20FPS 来研究处理的时间

+ 之后需要运行mesh分割方法建立反射平面。
+ 处理Pointcloud Data的时候为了方便人来看，需要安装open3d依赖. points点云始终返回一个大小为230400的3维数组。大约210000个元素是非零的。
+ mesh处理可以依托open3d来实现。

存储pointcloud的例程参考：https://docs.luxonis.com/software/depthai/examples/pointcloud_visualization/

## task2
+ 光线追踪算法和实现是最重要的。需要一些光线追踪的库来实现。candidate： pyoptix(nv),
+ 但是我们需要实现的功能与其叫光线追踪（Ray Tracing），不如叫做射线检测（Ray Castling）




### 现象
对于光线不佳、缺少纹理的表面数值会跳动较大

