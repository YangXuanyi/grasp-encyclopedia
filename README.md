# grasp-encyclopedia

![抓取过程图](./imgs/多目标抓取过程示意图.svg)

A benchmark platform for robot grasp detection, integrating some classic grasp algorithms.
# 项目简介
Grasp-encyclopedia 致力于打造一个抓取算法的通用平台，该平台将整理并集成历年来抓取领域的里程碑式开源算法。并以该领域几大著名数据集为基础，将算法封装在基于数据集的train-test程序框架中。为需要快速对比不同算法在同一数据集下性能的朋友提供帮助。


# 目录
[grasp-encyclopedia](#grasp-encyclopedia)
- [grasp-encyclopedia](#grasp-encyclopedia)
- [项目简介](#项目简介)
- [目录](#目录)
  - [Review](#review)
  - [2D-Grasp](#2d-grasp)
  - [6DoF-Grasp](#6dof-grasp)
  - [Datasets](#datasets)

## [Review](#Review)
该部分整理了抓取领域的相关综述论文，通过阅读这些论文可以迅速了解抓取领域的发展历程和最新进展。


|论文名称|    内容简介    |发表年份|文章链接|
|---|---|---|---|
|Vision-based Robotic Grasping From Object Localization, Object Pose Estimation to Grasp Estimation for Parallel Grippers: A Review|文章总结了基于视觉的机器人抓取过程中的三个关键任务：对象定位、对象姿态估计和抓取估计，基于这三项任务可以实现物体的2D平面抓取和6DoF抓取。此外总结了基于RGB-D图像输入的传统方法和最新的基于深度学习的方法；相关数据集和最先进方法之间的比较；基于视觉的机器人抓取面临的挑战以及解决这些挑战的未来方向。**是机器人抓取的入门综述论文**|2019|[arXiv](https://arxiv.org/abs/1905.06658)|
|Deep Learning Approaches to Grasp Synthesis: A Review|文章**针对性的梳理了6DoF抓取的发展状况**，并总结四种常见方法：基于采样的方法、直接回归、强化学习和示例方法。|2022|[arXiv](https://arxiv.org/abs/2207.02556)|
|Robotic Grasping from Classical to Modern: A Survey|文章回顾了基于分析和基于学习的机器人抓取方法。亮点是**对于抓取方法的分类比较细致，可以从文章的标题出发快速定位到基于某类技术实现的抓取方法**（例如：平面抓取中基于像素级表示的抓取方法）。|2022|[arXiv](https://arxiv.org/abs/2202.03631)|


## [2D-Grasp](#2D-Grasp)
2D 平面抓取是指目标物体位于平面工作空间上并且机械臂执行自上而下的抓取，这种抓取模式通常被称为 Top-down 抓取。在这种情况下，夹具的初始高度是固定的，并且夹具垂直于抓取平面。因此，抓取信息可以从 6D 简化为 3D，即2D 面内位置和 1D 旋转角度。

相关方法
## [6DoF-Grasp](#6DoF-Grasp)

## [Datasets](#Datasets)