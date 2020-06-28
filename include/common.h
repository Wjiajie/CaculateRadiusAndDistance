#ifndef COMMON_H
#define COMMON_H

#include "head.h"

struct Config
{
    bool m_visualization;
    bool m_caculate_distance;
    bool m_fit_cylinder_radius;
    double m_cluster_threshold;
    double m_pcl_fit_distance;

    //pixel
    int m_temp_width_pixel;
    int m_temp_l_height_pixel;

    //mm
    double m_template_width_mm ;
    double m_template_height_mm;

    Config(){};
    Config(const bool visualization, const bool caculate_distance, const bool fit_cylinder_radius, const double cluster_threshold,  double pcl_fit_distance, const int temp_width_pixel, const int temp_l_height_pixel, const double template_width_mm, const double template_height_mm)
    {
        this->m_visualization = visualization;
        this->m_caculate_distance = caculate_distance;
        this->m_fit_cylinder_radius = fit_cylinder_radius;
        this->m_cluster_threshold = cluster_threshold;
        this->m_pcl_fit_distance = pcl_fit_distance;
        this->m_temp_width_pixel = temp_width_pixel;
        this->m_temp_l_height_pixel = temp_l_height_pixel;
        this->m_template_width_mm = template_width_mm;
        this->m_template_height_mm = template_height_mm;
    }
};

struct BinocularCameraPara
{
    Mat m_cameraMatrix_l;
    Mat m_distCoeffs_l;
    Mat m_cameraMatrix_r;
    Mat m_distCoeffs_r;

    Mat m_R_relative;
    Mat m_t_relative;
    Mat m_E;
    Mat m_F;

    BinocularCameraPara(){};
    BinocularCameraPara(Mat cameraMatrix_l, Mat distCoeffs_l, Mat cameraMatrix_r, Mat distCoeffs_r, Mat R_relative, Mat t_relative, Mat E, Mat F)
    {
        m_cameraMatrix_l = cameraMatrix_l;
        m_distCoeffs_l = distCoeffs_l;
        m_cameraMatrix_r = cameraMatrix_r;
        m_distCoeffs_r = distCoeffs_r;

        m_R_relative = R_relative;
        m_t_relative = t_relative;
        m_E = E;
        m_F = F;
    }

};

struct View
{
    //poses
    Eigen::Matrix3d rotation;
    Eigen::Matrix3d K;
    Eigen::Matrix3d K_Inv;
    std::vector<double> distortionParams;
    Eigen::Matrix3d t_;
    Eigen::Vector3d t;

    double* rotation_array; //ÖáœÇ
    double* translatrion_array;
    double scale;
    View() {}
    View(Eigen::Matrix3d r, Eigen::Vector3d _t, Eigen::Matrix3d _K, std::vector<double> _distort)
    {
        this->rotation = r;
        this->t = _t;
        this->K = _K;
        this->distortionParams = _distort;

        this->K_Inv = this->K.inverse();

        t_ << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;

        this->rotation_array = new double[3];
        Eigen::AngleAxisd aa(this->rotation);
        Eigen::Vector3d v = aa.angle() * aa.axis();
        rotation_array[0] = v.x();
        rotation_array[1] = v.y();
        rotation_array[2] = v.z();

        this->translatrion_array = new double[3];
        translatrion_array[0] = t.x();
        translatrion_array[1] = t.y();
        translatrion_array[2] = t.z();
        this->scale = 1.0f;   //ºÍworldµÄ³ß¶È
    }
};

struct Observation
{
    Eigen::Vector2d pixel;
    Eigen::Vector2d match_pixel;
    int host_camera_id; //ÊôÓÚÄÄÒ»žöcamera
    int neighbor_camera_id; //ÓëÖ®Æ¥ÅäµÄcamera£¬ÓÃÀŽÑ°ÕÒÆ¥Åäµã


    Observation() {}
    Observation(Eigen::Vector2d p, Eigen::Vector2d m_p,int h_camera_id_ = -1, int n_camera_id_ = -1)
    {
        pixel = p;
        match_pixel = m_p;
        host_camera_id = h_camera_id_;
        neighbor_camera_id = n_camera_id_;
    }
};

struct Structure
{
    Eigen::Vector3d position;
    Eigen::Vector3d colors; //rgb
    uint structure_index;
    double* positions_array;
    bool isvalid_structure;

    Structure() {}
    Structure(Eigen::Vector3d _p , uint structure_index_,bool isvalid_structure_ = true)
    {
        position = _p;
        colors = Eigen::Vector3d::Zero();
        positions_array = new double[3];
        positions_array[0] = _p.x();
        positions_array[1] = _p.y();
        positions_array[2] = _p.z();

        structure_index = structure_index_;

        isvalid_structure = isvalid_structure_;
    }
};

//store the imformation of a feature in image
struct FeatureInImage
{
    KeyPoint m_feature;
    KeyPoint m_feature_match;
    Vector2d m_kp;
    Vector2d m_kp_match;
    Vector2d m_distance_from_edge;
    int m_which_temp;

    FeatureInImage(){};
    FeatureInImage(KeyPoint feature, KeyPoint feature_match, Vector2d kp, Vector2d kp_match, Vector2d distance_from_edge, int which_temp)
    {
        this->m_feature = feature;
        this->m_feature_match = feature_match;
        this->m_kp = kp;
        this->m_kp_match = kp_match;
        this->m_distance_from_edge = distance_from_edge;
        this->m_which_temp = which_temp;
    }
};

//BA
struct Ceres_Triangulate
{
    //2d µã
    const Eigen::Vector2d x;
    const Eigen::Matrix<double, 3, 4> P;  //Í¶Ó°ŸØÕó

    Ceres_Triangulate(Eigen::Vector2d x_, Eigen::Matrix<double, 3, 4> P_) :x(x_), P(P_) {}

    template<typename T>
    bool operator()(const T* const ceres_X, T* residual) const
    {

        T PX0 = T(P(0, 0)) * ceres_X[0] + T(P(0, 1)) * ceres_X[1] + T(P(0, 2)) * ceres_X[2] + T(P(0, 3));
        T PX1 = T(P(1, 0)) * ceres_X[0] + T(P(1, 1)) * ceres_X[1] + T(P(1, 2)) * ceres_X[2] + T(P(1, 3));
        T PX2 = T(P(2, 0)) * ceres_X[0] + T(P(2, 1)) * ceres_X[1] + T(P(2, 2)) * ceres_X[2] + T(P(2, 3));

        PX0 = PX0 / PX2;
        PX1 = PX1 / PX2;

        residual[0] = T(x.x()) - PX0;
        residual[1] = T(x.y()) - PX1;

        return true;
    }

};

struct Ceres_Triangulate_AdjustRt
{
    const Eigen::Vector2d x;
    const Eigen::Matrix<double, 3, 3> K;  //Í¶Ó°ŸØÕó


    Ceres_Triangulate_AdjustRt(Eigen::Vector2d x_, Eigen::Matrix<double, 3, 3> K_ ) :x(x_), K(K_){}

    template<typename T>
    bool operator()(const T* const ceres_X, const T* const ceres_angleAxis, const T* const ceres_t, T* residual) const
    {
        T PX[3];
        PX[0] = ceres_X[0];
        PX[1] = ceres_X[1];
        PX[2] = ceres_X[2];

        T PX_r[3];
        ceres::AngleAxisRotatePoint(ceres_angleAxis, PX, PX_r);

        T PX0 = T(K(0, 0)) * (PX_r[0] + ceres_t[0]) + T(K(0, 2)) * (PX_r[2] + ceres_t[2]);
        T PX1 = T(K(1, 1)) * (PX_r[1] + ceres_t[1]) + T(K(1, 2)) * (PX_r[2] + ceres_t[2]);
        T PX2 = (PX_r[2] + ceres_t[2]);

        PX0 = PX0 / PX2;
        PX1 = PX1 / PX2;


        residual[0] = T(x.x()) -  PX0;
        residual[1] = T(x.y()) -  PX1;
        return true;
    }

};

//BA
struct Ceres_refine_radius
{
    const Eigen::Vector3d position;
    const vector<double> para_array_fixed;

    Ceres_refine_radius(Eigen::Vector3d p_, vector<double> para_fix) :position(p_), para_array_fixed(para_fix) {}

    template<typename T>
    bool operator()(const T* const ceres_cylinder_para, T* residual) const
    {
        residual[0] = (T(position.x())-T(para_array_fixed[3])) * (T(position.x())-T(para_array_fixed[3]))
                + (T(position.y())-T(para_array_fixed[4])) * (T(position.y())-T(para_array_fixed[4]))
                + (T(position.z())-T(para_array_fixed[5])) * (T(position.z())-T(para_array_fixed[5]))
                - T(para_array_fixed[0]) * (T(position.x())-T(para_array_fixed[3]))
                - T(para_array_fixed[1]) * (T(position.y())-T(para_array_fixed[4]))
                - T(para_array_fixed[2]) * (T(position.z())-T(para_array_fixed[5]))
                - T(ceres_cylinder_para[0]) * T(ceres_cylinder_para[0]);

        return true;
    }

};

class Common
{
public:
    Common(void){};    
    void ReadBibocularCameraPara(string path, BinocularCameraPara & bino_cam);
    void ImageDedistortion(Mat src, Mat & dst , BinocularCameraPara bino_cam, int flag);    
    void FindMatchPoint(string intput_image_l_path, string intput_image_r_path, vector<Rect> temp_vec, vector<FeatureInImage> & features_in_image, Config cfg);
    void SavePLYFile_PointCloud(std::string filePath, std::vector<Structure> structure);
    void CalculateStructure_Init_DLT(vector<View> view, vector<Observation> obs , vector<Structure> & P );
    void CalculateStructure_Ceres(vector<View> view, vector<Observation> obs, vector<Structure> & output_struct);    
    void DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2d>points1, vector<Point2d>points2, cv::Mat F,string save_path);
    void FindLocalMatchRect(string feature_image,string intput_image, Rect & rect_local, Config cfg);
    void SelectFeature(Mat src_image_l, Mat temp_image_l, Mat temp_image_r, vector<FeatureInImage> features_in_image, vector<FeatureInImage> & features_selected, Config cfg);
    void CalculateDistance(vector<FeatureInImage> features_selected, vector<Structure> P, double radius);
    void Cylinderfitting(vector<Structure> P, double & ridius); //fit cylinder using MSE
    void PclCaculateRadius(vector<Structure> P, double & ridius, Config cfg); //fit cylylinder using Ransac
    ~Common(void){};
};

void fitCylinder(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out, double &radius, Config cfg);

Eigen::MatrixXd toEigenMatrixXd(const cv::Mat &cvMat);
vector<double> toStdVector(const cv::Mat &cvMat);

bool sort_by_vec_x(Point2d p1 ,Point2d p2);
bool sort_by_vec_y(Point2d p1 ,Point2d p2);
bool sort_by_vec_size(vector<Point2d> p1 ,vector<Point2d> p2);
void MaxMinCluster(vector<cv::Point2d> data,vector<vector<cv::Point2d>> & result, Config cfg);
// 计算两个模式样本之间的欧式距离
double get_distance(cv::Point2d data1, cv::Point2d data2);
// 寻找Z2,并计算阈值T
double step2(vector<cv::Point2d> data,double t, vector<cv::Point2d> & cluster_center);
void get_clusters(vector<cv::Point2d> data,vector<cv::Point2d> & cluster_center, double T);
//最近邻分类（离哪个聚点中心最近，归为哪类）
vector<vector<cv::Point2d>> classify(vector<cv::Point2d> data, vector<cv::Point2d> & cluster_center, double T);

#endif // COMMON_H
