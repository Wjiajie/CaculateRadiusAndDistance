#include "caculateridiusanddistance.h"
const string path = "/home/jiajie/3d_reco/Binocular/final-result/CaculateObjectLength/featureImage";
const bool visualization = true;
const bool caculate_distance = true;
const bool fit_cylinder_radius = true;
const double cluster_threshold = 1.0;
const double pcl_fit_distance = 0.5;

//the template size in "pixel" count
const int temp_width_pixel = 76;
const int temp_height_pixel = 378;

//the template size in "millimeters" count
const double template_width_mm =  20.0;
const double template_height_mm = 99.86;

int main(void)
{   
    Config cfg(visualization, caculate_distance, fit_cylinder_radius, cluster_threshold, pcl_fit_distance, temp_width_pixel, temp_height_pixel, template_width_mm, template_height_mm);
    const string intput_image_l = path + "/snap_venc2_4.jpg";
    const string intput_image_r = path + "/snap_venc3_4.jpg";
    const string feature_image_l = path +  "/feature1.png";
    const string feature_image_r = path + "/feature2.png";
    caculateRadiusAndDistance(path, intput_image_l, intput_image_r, feature_image_l, feature_image_r, cfg);
	return 0;
}
