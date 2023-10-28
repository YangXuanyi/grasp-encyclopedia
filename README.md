# grasp-encyclopedia

![绝对路径的图片](imgs\多目标抓取过程示意图.svg)

A benchmark platform for robot grasp detection, integrating some classic grasp algorithms.

#
# 项目简介
Grasp-encyclopedia 致力于打造一个抓取算法的通用平台，该平台将整理并集成历年来抓取领域的里程碑式开源算法。并以该领域几大著名数据集为基础，将算法封装在基于数据集的train-test程序框架中。为需要快速对比不同算法在同一数据集下性能的朋友提供帮助。
#


# 目录
* [Review](#Review)
* [2D-Grasp](#2D-Grasp)
* [6DoF-Grasp](#6DoF-Grasp)
* [Datasets](#Datasets)

## [Review](#Review)
该部分整理了抓取领域的相关综述论文，通过阅读这些论文可以迅速了解抓取领域的发展历程和最新进展。


|论文名称|    内容简介    |发表年份|文章链接|
|---|---|---|---|
|Vision-based Robotic Grasping From Object Localization, Object Pose Estimation to Grasp Estimation for Parallel Grippers: A Review|文章总结了基于视觉的机器人抓取过程中的三个关键任务：对象定位、对象姿态估计和抓取估计，基于这三项任务可以实现物体的2D平面抓取和6DoF抓取。此外总结了基于RGB-D图像输入的传统方法和最新的基于深度学习的方法；相关数据集和最先进方法之间的比较；基于视觉的机器人抓取面临的挑战以及解决这些挑战的未来方向。**是机器人抓取的入门综述论文**|2019|[arXiv](https://arxiv.org/abs/1905.06658)|

