#include "caculateridiusanddistance.h"

void caculateRadiusAndDistance(const string path, string intput_image_l, string intput_image_r, string feature_image_l, string feature_image_r, Config cfg)
{
    Common common;

    //Read the camera para and raw data,

    BinocularCameraPara bino_cam;
    string para_path = path + "/intrinsics.yml";
    common.ReadBibocularCameraPara(para_path, bino_cam);

    Mat src_image_l_raw = imread(intput_image_l);
    Mat src_image_r_raw = imread(intput_image_r);

    Mat src_image_l;
    Mat src_image_r;

    //ImageDedistortion
    common.ImageDedistortion(src_image_l_raw, src_image_l, bino_cam, 0);
    common.ImageDedistortion(src_image_r_raw, src_image_r, bino_cam, 1);

    Mat temp_image_l = imread(feature_image_l);
    Mat temp_image_r = imread(feature_image_r);

    Rect rect_temp_l_src_l;
    Rect rect_temp_r_src_l;
    Rect rect_temp_l_src_r;
    Rect rect_temp_r_src_r;

    //input: feature template and src image ,output: the local interest area bbox ,no need to return local feature and match relationsive
    common.FindLocalMatchRect(feature_image_l,intput_image_l,rect_temp_l_src_l, cfg);
    cout<<"rect_feature_l_src_l is: "<<rect_temp_l_src_l<<endl;
    common.FindLocalMatchRect(feature_image_r,intput_image_l,rect_temp_r_src_l, cfg);
    cout<<"rect_feature_r_src_l is: "<<rect_temp_r_src_l<<endl;
    //input: feature template and src image ,output: the local interest area bbox ,no need to return local feature and match relationsive
    common.FindLocalMatchRect(feature_image_l,intput_image_r,rect_temp_l_src_r, cfg);
    cout<<"rect_feature_l_src_r is: "<<rect_temp_l_src_r<<endl;
    common.FindLocalMatchRect(feature_image_r,intput_image_r,rect_temp_r_src_r, cfg);
    cout<<"rect_feature_r_src_r is: "<<rect_temp_r_src_r<<endl;
    vector<Rect> temp_vec{rect_temp_l_src_l, rect_temp_r_src_l, rect_temp_l_src_r, rect_temp_r_src_r };

    //test
    Mat img_display_l = imread(intput_image_l);
    rectangle(img_display_l, rect_temp_l_src_l, Scalar(0,0,255), 2, 8, 0 );
    rectangle(img_display_l, rect_temp_r_src_l, Scalar(0,0,255), 2, 8, 0 );
    Mat img_display_r = imread(intput_image_r);
    rectangle(img_display_r, rect_temp_l_src_r, Scalar(0,0,255), 2, 8, 0 );
    rectangle(img_display_r, rect_temp_r_src_r, Scalar(0,0,255), 2, 8, 0 );
    if(cfg.m_visualization)
    {
        imwrite(path + "/result_l.jpg",img_display_l);
        imwrite(path + "/result_r.jpg",img_display_r);
    }
    vector<FeatureInImage> features_in_image;
    common.FindMatchPoint(intput_image_l, intput_image_r, temp_vec, features_in_image, cfg);
    vector<FeatureInImage> features_selected;
    //select pairs of feature most close to the center of circle
    if(cfg.m_caculate_distance)
    {
        common.SelectFeature(src_image_l, temp_image_l, temp_image_r,  features_in_image, features_selected, cfg);
        cout<<"features_selected size: "<<features_selected.size()<<endl;
    }

    Matrix3d R0;
    R0 << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    Vector3d t0;
    t0 << 0, 0, 0;

    cv::Mat_<double> a1 = Mat_<double>::ones(2,2);

    Eigen::Matrix3d K0;
    K0 = toEigenMatrixXd(bino_cam.m_cameraMatrix_l);

    vector<double> distort0;
    distort0 = toStdVector(bino_cam.m_distCoeffs_l);
    View view0(R0, t0, K0, distort0);

    Eigen::Matrix3d R1;
    R1  = toEigenMatrixXd(bino_cam.m_R_relative);

    Eigen::Vector3d t1;
    t1  = toEigenMatrixXd(bino_cam.m_t_relative);

    Eigen::Matrix3d K1;
    K1 = toEigenMatrixXd(bino_cam.m_cameraMatrix_r);

    std::vector<double> distort1;
    distort1 = toStdVector(bino_cam.m_distCoeffs_r);

    View view1(R1, t1, K1, distort1);

    vector<View> view{ view0,view1 };

    for(auto & v:view)
    {
        cout<<v.K<<endl<<v.rotation<<endl<<v.t<<endl<<endl;
    }

    //fit the Cylinder
    vector<Observation> obs;
    vector<Structure> P;
    double radius = 35.0;
    if(cfg.m_fit_cylinder_radius)
    {
        cout<<"features_in_image size: "<<features_in_image.size()<<endl;
        for (int j = 0; j < features_in_image.size(); ++j)
        {
            obs.emplace_back(Observation(features_in_image[j].m_kp, features_in_image[j].m_kp_match, 0 , 1 ));
        }
        common.CalculateStructure_Init_DLT(view, obs, P);
        if(cfg.m_visualization)
        {
            common.SavePLYFile_PointCloud(path + "/test.ply", P);
        }
        common.CalculateStructure_Ceres(view, obs, P);
        if(cfg.m_visualization)
        {
            common.SavePLYFile_PointCloud(path + "/test_ceres.ply", P);
        }
        common.PclCaculateRadius(P, radius, cfg);
        //caculate the height of Cylinder
        obs.clear();
        P.clear();
    }

    if(cfg.m_caculate_distance)
    {
        for (int j = 0; j < features_selected.size(); ++j)
        {
            obs.emplace_back(Observation(features_selected[j].m_kp, features_selected[j].m_kp_match, 0 , 1 ));
        }
        common.CalculateStructure_Init_DLT(view, obs, P);
        common.CalculateStructure_Ceres(view, obs, P);
        common.CalculateDistance(features_selected, P, radius);
    }
    if(cfg.m_fit_cylinder_radius)
    {
        cout<<"cylinder radius is : "<<radius<<" mm"<<endl;
    }
}
