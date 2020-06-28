# CaculateRadiusAndDistance
基于模板匹配的方法，用双目相机测量圆柱体的两端距离，示意图：

<img src="https://i.loli.net/2020/06/28/gSNCtQ68vkU4sJ2.png" width = "80%" height = "40%" div align = center />

重建模板点云：

<img src="https://i.loli.net/2020/06/28/MX2kJgUy7jzZWF6.png" width = "80%" height = "40%" div align = center />



依赖环境：

```c++
ubuntu 18.04
opencv 3.4.3
opencv_contrib 3.4.3 (为了使用sift)
eigen 3.2.10
ceres-solover 1.13.0
pcl-1.9.1
```

