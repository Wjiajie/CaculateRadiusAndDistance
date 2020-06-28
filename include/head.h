#ifndef HEAD_H
#define HEAD_H

#include <vector>
#include <iostream>
#include <iterator>
#include <random>
#include <math.h>
//#include "cv.h"
//#include <cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<opencv2/features2d/features2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

//pcl
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/segmentation/sac_segmentation.h>
#include<pcl/search/search.h>
#include<pcl/search/kdtree.h>
#include<pcl/features/normal_3d.h>
#include<pcl/common/common.h>


using namespace std;
using namespace cv;
using namespace Eigen;


#endif // HEAD_H
