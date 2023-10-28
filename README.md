# <p align=center>grasp-encyclopedia</p>

![抓取过程图](./imgs/多目标抓取过程示意图.svg)

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat#pic_center)](http://makeapullrequest.com)

A benchmark platform for robot grasp detection, integrating some classic grasp algorithms.


- [grasp-encyclopedia](#grasp-encyclopedia)
- [项目简介](#项目简介)
- [Review](#review)
- [2D-Grasp](#2d-grasp)
- [6DoF-Grasp](#6dof-grasp)
- [Datasets](#datasets)

# 项目简介
Grasp-encyclopedia 致力于打造一个抓取算法的通用平台，该平台将整理并集成历年来抓取领域的里程碑式开源算法。并以该领域几大著名数据集为基础，将算法封装在基于数据集的train-test程序框架中。为需要快速对比不同算法在同一数据集下性能的朋友提供帮助。

# [Review](#Review)
该部分整理了抓取领域的相关综述论文，通过阅读这些论文可以迅速了解抓取领域的发展历程和最新进展。

|论文名称|    内容简介    |发表年份|文章链接|
|---|---|---|---|
|Vision-based Robotic Grasping From Object Localization, Object Pose Estimation to Grasp Estimation for Parallel Grippers: A Review|文章总结了基于视觉的机器人抓取过程中的三个关键任务：对象定位、对象姿态估计和抓取估计，基于这三项任务可以实现物体的2D平面抓取和6DoF抓取。此外总结了基于RGB-D图像输入的传统方法和最新的基于深度学习的方法；相关数据集和最先进方法之间的比较；基于视觉的机器人抓取面临的挑战以及解决这些挑战的未来方向。**是机器人抓取的入门综述论文**|2019|[arXiv](https://arxiv.org/abs/1905.06658)|
|Deep Learning Approaches to Grasp Synthesis: A Review|文章**针对性的梳理了6DoF抓取的发展状况**，并总结四种常见方法：基于采样的方法、直接回归、强化学习和示例方法。|2022|[arXiv](https://arxiv.org/abs/2207.02556)|
|Robotic Grasping from Classical to Modern: A Survey|文章回顾了基于分析和基于学习的机器人抓取方法。亮点是**对于抓取方法的分类比较细致，可以从文章的标题出发快速定位到基于某类技术实现的抓取方法**（例如：平面抓取中基于像素级表示的抓取方法）。|2022|[arXiv](https://arxiv.org/abs/2202.03631)|


# [2D-Grasp](#2D-Grasp)
2D 平面抓取是指目标物体位于平面工作空间上并且机械臂执行自上而下的抓取，这种抓取模式通常被称为 Top-down 抓取。在这种情况下，夹具的初始高度是固定的，并且夹具垂直于抓取平面。因此，抓取信息可以从 6D 简化为 3D，即2D 面内位置和 1D 旋转角度。

|算法简称|论文名称|    算法简介    |输入数据|发表刊物及时间|
|---|---|---|---|---|
|---|Real-Time Grasp Detection Using Convolutional Neural Networks [[paper]](https://arxiv.org/abs/1412.3128), [[code]](https://github.com/tnikolla/robot-grasp-detection)|文章提出了一种基于CNN网络的实时抓取检测方法，将抓取检测当作一阶段目标检测任务进行回归检测。该论文是第一个提出使用神经网络实现抓取检测，作者是CV领域大名鼎鼎的Joseph Redmon。|RGB|ICRA 2015|
|GG-CNN|Learning robust, real-time, reactive robotic grasping [[paper]](https://journals.sagepub.com/doi/full/10.1177/0278364919859066), [[code]](https://github.com/dougsm/ggcnn)|文章提出了一种在深度图上实现像素级抓取检测的方法，轻量化。|D|Robotics: Science and Systems (RSS) 2018|
|GRCNN|Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network [[paper]](https://arxiv.org/abs/1909.04810), [[code]](https://github.com/skumra/robotic-grasping)|文章提出了一种生成残差卷积神经网络(GR ConvNet)模型，使用像素级表示方法进行检测，输出角度、宽度、质量热图矩阵。论文很好的平衡了速度和精度问题，同时给出了机械臂驱动代码，部署友好。|RGB-D|IROS 2020|
|E2E-net|End-to-end Trainable Deep Neural Network for Robotic Grasp Detection and Semantic Segmentation from RGB [[paper]](https://arxiv.org/abs/2107.05287), [[code]](https://github.com/stefan-ainetter/grasp_det_seg_cnn)|文章引入了一种细化模块，实现了抓取检测和语义分割，同时扩展了ORCID数据集。在CNN类型的开源算法里精度比较高。|RGB|ICRA 2021|
|TF-Grasp|When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection [[paper]](https://ieeexplore.ieee.org/abstract/document/9810182), [[code]](https://github.com/WangShaoSUN/grasp-transformer)|第一个使用transformer模型实现像素级抓取检测。|RGB-D|IEEE Robotics and Automation Letters 2022|
|SSG|Instance-wise Grasp Synthesis for Robotic Grasping [[paper]](https://arxiv.org/abs/2302.07824), [[code]](https://github.com/HilbertXu/Instance-wise-grasp-synthesis)|文章提出一种单阶段的像素级抓取网络，同时为每个对象生成实例掩码和抓取配置。|RGB-D|ICRA 2023|

# [6DoF-Grasp](#6DoF-Grasp)
xxxx

# [Datasets](#Datasets)

|名称|简介|应用领域|
|---|---|---|
|[Cornell Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)|平面抓取检测领域中最为常见的数据集之一，由1035 张 RGB-D 图像组成，其中每张图像的分辨率为 640*480，数据集中包含了 240 个不同的物体。|平面抓取|
|[Jacquard Dataset](https://jacquard.liris.cnrs.fr/)|一个典型的2D抓取检测数据集，内部包含54k张GRB-D图像和110万个抓取实例。|平面抓取|