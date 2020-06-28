#include "common.h"

const int SELECT_NUM = 30;
const double DISTANCE_TRESHOLD = 1;
int globel_count = 0;

const double pi = 3.14159;

const double MATCH_THRESHOLD = 1;

const int64  MAX_POINT_NUM = 10000;

void Common::ReadBibocularCameraPara(string path, BinocularCameraPara & bino_cam)
{
    bool FSflag = false;
    FileStorage readfs;

    FSflag = readfs.open(path, FileStorage::READ);
    if (FSflag == false) cout << "Cannot open the file" << endl;
    readfs["M1"] >> bino_cam.m_cameraMatrix_l;
    readfs["D1"] >> bino_cam.m_distCoeffs_l;
    readfs["M2"] >> bino_cam.m_cameraMatrix_r;
    readfs["D2"] >> bino_cam.m_distCoeffs_r;
    readfs["R"] >> bino_cam.m_R_relative;
    readfs["T"] >> bino_cam.m_t_relative;
    readfs["E"] >> bino_cam.m_E;
    readfs["F"] >> bino_cam.m_F;

    readfs.release();

}

void Common::ImageDedistortion(Mat src, Mat & dst , BinocularCameraPara bino_cam, int flag)
{
    Mat output;
    Mat	cameraMatrix, distCoeffs;
    Size imageSize;
    Mat map1, map2;

    if(flag)
    {
        cameraMatrix = bino_cam.m_cameraMatrix_r.clone();
        distCoeffs = bino_cam.m_distCoeffs_r.clone();
    }
    else {
        cameraMatrix = bino_cam.m_cameraMatrix_l.clone();
        distCoeffs = bino_cam.m_distCoeffs_l.clone();
    }

    cout<<"cameraMatrix: "<<cameraMatrix<<endl;
    cout<<"distCoeffs: "<<distCoeffs<<endl;

    Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0);   
    undistort(src, dst, cameraMatrix, distCoeffs);

}

void Common::FindMatchPoint(string intput_image_l_path, string intput_image_r_path, vector<Rect> temp_vec, vector<FeatureInImage> & features_in_image, Config cfg)
{
    Mat src_image_l = imread(intput_image_l_path);
    Mat src_image_r = imread(intput_image_r_path);

    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector>  sift_detector = xfeatures2d::SiftFeatureDetector::create();

    std::vector<KeyPoint> key_points1;
    std::vector<KeyPoint> key_points2;

    Mat imageDesc1, imageDesc2;
    sift_detector->detectAndCompute(src_image_l, Mat(), key_points1, imageDesc1);
    sift_detector->detectAndCompute(src_image_r, Mat(), key_points2, imageDesc2);

    //获取匹配特征点，提取最优匹配
    FlannBasedMatcher matcher;
    vector<DMatch> matchPoints, matchPoints_select;
    vector<int> temp_index;
    matcher.match(imageDesc1, imageDesc2, matchPoints, Mat());
    sort(matchPoints.begin(), matchPoints.end());//特征排序

    vector<Vector2d> imagePoints1_init;
    vector<Vector2d> imagePoints2_init;

    //if kps in temp_l
    for (int i = 0; i < matchPoints.size(); i++)
    {
        if(key_points1[matchPoints[i].queryIdx].pt.x > temp_vec[0].x && key_points1[matchPoints[i].queryIdx].pt.y > temp_vec[0].y)
        {
            if(key_points1[matchPoints[i].queryIdx].pt.x < (temp_vec[0].x + temp_vec[0].width) && key_points1[matchPoints[i].queryIdx].pt.y < (temp_vec[0].y + temp_vec[0].height))
            {
                if(key_points2[matchPoints[i].trainIdx].pt.x > temp_vec[2].x && key_points2[matchPoints[i].trainIdx].pt.y > temp_vec[2].y)
                {
                    if(key_points2[matchPoints[i].trainIdx].pt.x < (temp_vec[2].x + temp_vec[2].width) && key_points2[matchPoints[i].trainIdx].pt.y < (temp_vec[2].y + temp_vec[2].height))
                    {
                        temp_index.push_back(0);
                        matchPoints_select.emplace_back(matchPoints[i]);
                        imagePoints1_init.push_back(Vector2d(key_points1[matchPoints[i].queryIdx].pt.x,key_points1[matchPoints[i].queryIdx].pt.y));
                        imagePoints2_init.push_back(Vector2d(key_points2[matchPoints[i].trainIdx].pt.x,key_points2[matchPoints[i].trainIdx].pt.y));
                    }
                }

            }
        }

    }

    //if kps in temp_r
    if(cfg.m_caculate_distance)
    {
        for (int i = 0; i < matchPoints.size(); i++)
        {
            if(key_points1[matchPoints[i].queryIdx].pt.x > temp_vec[1].x && key_points1[matchPoints[i].queryIdx].pt.y > temp_vec[1].y)
            {
                if(key_points1[matchPoints[i].queryIdx].pt.x < (temp_vec[1].x + temp_vec[1].width) && key_points1[matchPoints[i].queryIdx].pt.y < (temp_vec[1].y + temp_vec[1].height))
                {
                    if(key_points2[matchPoints[i].trainIdx].pt.x > temp_vec[3].x && key_points2[matchPoints[i].trainIdx].pt.y > temp_vec[3].y)
                    {
                        if(key_points2[matchPoints[i].trainIdx].pt.x < (temp_vec[3].x + temp_vec[3].width) && key_points2[matchPoints[i].trainIdx].pt.y < (temp_vec[3].y + temp_vec[3].height))
                        {
                            temp_index.push_back(1);
                            matchPoints_select.emplace_back(matchPoints[i]);
                            imagePoints1_init.push_back(Vector2d(key_points1[matchPoints[i].queryIdx].pt.x,key_points1[matchPoints[i].queryIdx].pt.y));
                            imagePoints2_init.push_back(Vector2d(key_points2[matchPoints[i].trainIdx].pt.x,key_points2[matchPoints[i].trainIdx].pt.y));
                        }
                    }

                }
            }
        }
    }


    vector<Point2d> p1,p2;

    //ransac remove outliners
    for (int i = 0; i < imagePoints1_init.size(); i++)
    {
        p1.push_back(Point2d(imagePoints1_init[i][0],imagePoints1_init[i][1]));
    }
    for (int i = 0; i < imagePoints2_init.size(); i++)
    {
        p2.push_back(Point2d(imagePoints2_init[i][0],imagePoints2_init[i][1]));
    }

    vector<uchar> m_RANSACStatus;

    Mat F = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC,3.0,0.99);

    int index = 0;
    vector<KeyPoint> leftInlier;
    vector<KeyPoint> rightInlier;
    vector<DMatch> inlierMatch;

    for (int i = 0; i < matchPoints_select.size(); i++)
    {
        if(m_RANSACStatus[i] != 0)
        {
            auto ft1 = key_points1[matchPoints_select[i].queryIdx];
            auto ft2 = key_points2[matchPoints_select[i].trainIdx];
            leftInlier.emplace_back(ft1);
            rightInlier.emplace_back(ft2);
            matchPoints_select[i].queryIdx = index;
            matchPoints_select[i].trainIdx = index;
            inlierMatch.emplace_back(matchPoints_select[i]);
            index++;

            FeatureInImage Feature_in_image = FeatureInImage(ft1,ft2,Vector2d(ft1.pt.x,ft1.pt.y),Vector2d(ft2.pt.x,ft2.pt.y),Vector2d(-1,-1),temp_index[i]);
            features_in_image.emplace_back(Feature_in_image);
        }
    }


    //Mat image_match;

    //drawMatches(src_image_l, leftInlier, src_image_r, rightInlier, inlierMatch, image_match);
    //imwrite("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/featureImage/matched.png",image_match);
    //imwrite("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/featureImage/matched.png",image_match);

}

void Common::FindLocalMatchRect(string feature_image,string intput_image, Rect & rect_local, Config cfg)
{
    Mat src_image_l = imread(feature_image);
    Mat src_image_r = imread(intput_image);

    vector<Vector2d>  imagePoints1;
    vector<Vector2d>  imagePoints2;

    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector>  sift_detector = xfeatures2d::SiftFeatureDetector::create();

    std::vector<KeyPoint> key_points1;
    std::vector<KeyPoint> key_points2;

    Mat imageDesc1, imageDesc2;
    sift_detector->detectAndCompute(src_image_l, Mat(), key_points1, imageDesc1);
    sift_detector->detectAndCompute(src_image_r, Mat(), key_points2, imageDesc2);

    //获取匹配特征点，提取最优匹配
    FlannBasedMatcher matcher;
    vector<DMatch> matchPoints, matchPoints_select;
    matcher.match(imageDesc1, imageDesc2, matchPoints, Mat());
    sort(matchPoints.begin(), matchPoints.end());//特征排序

    const int N = matchPoints.size()>SELECT_NUM?SELECT_NUM:matchPoints.size();

    for (int i = 0; i < N; i++)
    {
        matchPoints_select.emplace_back(matchPoints[i]);
        imagePoints1.push_back(Vector2d(key_points1[matchPoints[i].queryIdx].pt.x,key_points1[matchPoints[i].queryIdx].pt.y));
        imagePoints2.push_back(Vector2d(key_points2[matchPoints[i].trainIdx].pt.x,key_points2[matchPoints[i].trainIdx].pt.y));
    }

    vector<Point2d> p1,p2;

    //ransac remove outliners
    for (int i = 0; i < imagePoints1.size(); i++)
    {
        p1.push_back(Point2d(imagePoints1[i][0],imagePoints1[i][1]));
    }
    for (int i = 0; i < imagePoints2.size(); i++)
    {
        p2.push_back(Point2d(imagePoints2[i][0],imagePoints2[i][1]));
    }

    vector<uchar> m_RANSACStatus;

    Mat F = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC, 3.0,0.99);

    int index = 0;
    vector<KeyPoint> leftInlier;
    vector<KeyPoint> rightInlier;
    vector<DMatch> inlierMatch;

    for (int i = 0; i < matchPoints_select.size(); i++)
    {
        //m_RANSACStatus[i] != 0
        if(m_RANSACStatus[i] != 0)
        {
            auto ft1 = key_points1[matchPoints_select[i].queryIdx];
            auto ft2 = key_points2[matchPoints_select[i].trainIdx];
            leftInlier.emplace_back(ft1);
            rightInlier.emplace_back(ft2);
            matchPoints_select[i].queryIdx = index;
            matchPoints_select[i].trainIdx = index;
            inlierMatch.emplace_back(matchPoints_select[i]);
            index++;
        }
    }

    Mat image_match;

    drawMatches(src_image_l, leftInlier, src_image_r, rightInlier, inlierMatch, image_match);
    imwrite("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/featureImage/match"+to_string(globel_count++)+".png",image_match);

    vector<cv::Point2d> keypoints_vec;

    for(int i = 0;i<rightInlier.size();++i)
    {
        keypoints_vec.emplace_back(rightInlier[i].pt);
    }


    vector<vector<Point2d>> keypoints_vec_cluster;
    MaxMinCluster(keypoints_vec,keypoints_vec_cluster, cfg);

    sort(keypoints_vec_cluster.begin(),keypoints_vec_cluster.end(),sort_by_vec_size);
    cout<<"keypoints_vec_cluster.size(): "<<keypoints_vec_cluster[0].size()<<endl;
    sort(keypoints_vec_cluster[0].begin(),keypoints_vec_cluster[0].end(),sort_by_vec_x);
    int expand_size = 1.0;
    int rect_x = int(keypoints_vec_cluster[0][0].x);
    int rect_x_max = int(keypoints_vec_cluster[0][keypoints_vec_cluster[0].size()-1].x);
    int feature_width = rect_x_max - rect_x;
    rect_x = int(rect_x - expand_size * feature_width) > 0?int(rect_x - expand_size * feature_width):0;
    rect_x_max = int(rect_x_max +  expand_size * feature_width)<int(src_image_r.cols)?int(rect_x_max +  expand_size * feature_width):int(src_image_r.cols);
    sort(keypoints_vec_cluster[0].begin(),keypoints_vec_cluster[0].end(),sort_by_vec_y);
    int rect_y = int(keypoints_vec_cluster[0][0].y);
    int rect_y_max = int(keypoints_vec_cluster[0][keypoints_vec_cluster[0].size()-1].y);
    int feature_height = rect_y_max - rect_y;
    rect_y = int(rect_y - expand_size * feature_height) > 0?int(rect_y - expand_size * feature_height):0;
    rect_y_max = int(rect_y_max +  expand_size * feature_height)<int(src_image_r.rows)?int(rect_y_max +  expand_size * feature_height):int(src_image_r.rows);
    rect_local = Rect(rect_x,rect_y,rect_x_max - rect_x,rect_y_max - rect_y);
}

void Common::SelectFeature(Mat src_image_l, Mat temp_image_l, Mat temp_image_r, vector<FeatureInImage> features_in_image, vector<FeatureInImage> & features_selected, Config cfg)
{
    features_selected.clear();
    //find match between stored feature and template feature
    vector<KeyPoint> key_points_src_l_temp_l;
    vector<KeyPoint> key_points_src_l_temp_r;
    for(int i = 0;i<features_in_image.size();++i)
    {
        if(features_in_image[i].m_which_temp == 0)
        {
            key_points_src_l_temp_l.emplace_back(features_in_image[i].m_feature);
        }
        else {
            key_points_src_l_temp_r.emplace_back(features_in_image[i].m_feature);
        }
    }


    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector>  sift_detector = xfeatures2d::SiftFeatureDetector::create();

    std::vector<KeyPoint> key_points_temp_l;
    std::vector<KeyPoint> key_points_temp_r;

    Mat imageDesc_src_l_temp_l, imageDesc_src_l_temp_r;
    sift_detector->compute(src_image_l, key_points_src_l_temp_l, imageDesc_src_l_temp_l);
    sift_detector->compute(src_image_l, key_points_src_l_temp_r, imageDesc_src_l_temp_r);

    Mat imageDesc_temp_l, imageDesc_temp_r;
    sift_detector->detectAndCompute(temp_image_l, Mat(), key_points_temp_l, imageDesc_temp_l);
    sift_detector->detectAndCompute(temp_image_r, Mat(), key_points_temp_r, imageDesc_temp_r);


    FlannBasedMatcher matcher;
    vector<DMatch> matchPoints_src_l_temp_l, matchPoints_src_l_temp_r;
    matcher.match(imageDesc_src_l_temp_l, imageDesc_temp_l, matchPoints_src_l_temp_l, Mat());
    sort(matchPoints_src_l_temp_l.begin(), matchPoints_src_l_temp_l.end());//特征排序

    //ransac remove outliners
    vector<Point2d> p1,p2;

    for (int i = 0; i < matchPoints_src_l_temp_l.size(); i++)
    {
        p1.push_back(key_points_src_l_temp_l[matchPoints_src_l_temp_l[i].queryIdx].pt);
        p2.push_back(key_points_temp_l[matchPoints_src_l_temp_l[i].trainIdx].pt);
    }

    vector<uchar> m_RANSACStatus1;

    Mat F1 = findFundamentalMat(p1, p2, m_RANSACStatus1, FM_RANSAC,1.0,0.99);

    int index1 = 0;
    vector<KeyPoint> leftInlier1;
    vector<KeyPoint> rightInlier1;
    vector<DMatch> inlierMatch1;
    vector<DMatch> inlierMatch1_sorted;

    for (int i = 0; i < matchPoints_src_l_temp_l.size(); i++)
    {
        //m_RANSACStatus1[i] != 0
        if(m_RANSACStatus1[i] != 0)
        {
            auto ft1 = key_points_src_l_temp_l[matchPoints_src_l_temp_l[i].queryIdx];
            auto ft2 = key_points_temp_l[matchPoints_src_l_temp_l[i].trainIdx];
            leftInlier1.emplace_back(ft1);
            rightInlier1.emplace_back(ft2);
            inlierMatch1.emplace_back(matchPoints_src_l_temp_l[i]);
            //need to test
            matchPoints_src_l_temp_l[i].queryIdx = index1;
            matchPoints_src_l_temp_l[i].trainIdx = index1;
            inlierMatch1_sorted.emplace_back(matchPoints_src_l_temp_l[i]);
            index1++;
        }
    }


    for (int i = 0; i < inlierMatch1_sorted.size(); i++)
    {               
        double x_bias = cfg.m_template_width_mm * rightInlier1[inlierMatch1_sorted[i].trainIdx].pt.x/(1.0 *cfg.m_temp_width_pixel);
        double y_bias = cfg.m_template_height_mm * rightInlier1[inlierMatch1_sorted[i].trainIdx].pt.y/(1.0 *cfg.m_temp_l_height_pixel);
        Vector2d temp_x_y = Vector2d(x_bias,y_bias);
        //cout<<"temp_x_y: "<<temp_x_y<<endl;
        features_in_image[inlierMatch1[i].queryIdx].m_distance_from_edge = temp_x_y;
        //cout<<"inlierMatch1[i].queryIdx: "<<inlierMatch1[i].queryIdx<<endl;
    }

    //draw picture
    //Mat image_match_l;
    //drawMatches(src_image_l, leftInlier1, temp_image_l, rightInlier1, inlierMatch1_sorted, image_match_l);
    //imwrite("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/featureImage/matched_src_l_temp_l.png",image_match_l);

//****************************


    matcher.match(imageDesc_src_l_temp_r, imageDesc_temp_r, matchPoints_src_l_temp_r, Mat());
    sort(matchPoints_src_l_temp_r.begin(), matchPoints_src_l_temp_r.end());//特征排序

    //ransac remove outliners
    vector<Point2d> p3,p4;

    for (int i = 0; i < matchPoints_src_l_temp_r.size(); i++)
    {
        p3.push_back(key_points_src_l_temp_r[matchPoints_src_l_temp_r[i].queryIdx].pt);
        p4.push_back(key_points_temp_r[matchPoints_src_l_temp_r[i].trainIdx].pt);
    }

    vector<uchar> m_RANSACStatus2;

    Mat F2 = findFundamentalMat(p3, p4, m_RANSACStatus2, FM_RANSAC,1.0,0.99);

    int index2 = 0;
    vector<KeyPoint> leftInlier2;
    vector<KeyPoint> rightInlier2;
    vector<DMatch> inlierMatch2;
    vector<DMatch> inlierMatch2_sorted;

    for (int i = 0; i < matchPoints_src_l_temp_r.size(); i++)
    {
        //m_RANSACStatus2[i] != 0
        if(m_RANSACStatus2[i] != 0)
        {
            auto ft3 = key_points_src_l_temp_r[matchPoints_src_l_temp_r[i].queryIdx];
            auto ft4 = key_points_temp_r[matchPoints_src_l_temp_r[i].trainIdx];
            leftInlier2.emplace_back(ft3);
            rightInlier2.emplace_back(ft4);
            inlierMatch2.emplace_back(matchPoints_src_l_temp_r[i]);
            matchPoints_src_l_temp_r[i].queryIdx = index2;
            matchPoints_src_l_temp_r[i].trainIdx = index2;
            inlierMatch2_sorted.emplace_back(matchPoints_src_l_temp_r[i]);
            index2++;
        }
    }


    vector<Vector2d>  imagePoints_src_l_temp_r;
    vector<Vector2d>  imagePoints_temp_r;


    for (int i = 0; i < inlierMatch2_sorted.size(); i++)
    {        
        //template width 20.5, template height 60.5
        double x_bias = cfg.m_template_width_mm * (1 - rightInlier2[inlierMatch2_sorted[i].trainIdx].pt.x/(1.0 *cfg.m_temp_width_pixel));
        double y_bias = cfg.m_template_height_mm * rightInlier2[inlierMatch2_sorted[i].trainIdx].pt.y/(1.0 *cfg.m_temp_l_height_pixel);
        Vector2d temp_x_y = Vector2d(x_bias,y_bias);
        //cout<<"temp_x_y: "<<temp_x_y<<endl;
        features_in_image[key_points_src_l_temp_l.size() + inlierMatch2[i].queryIdx].m_distance_from_edge = temp_x_y;
        //cout<<"inlierMatch2[i].queryIdx: "<<inlierMatch2[i].queryIdx<<endl;
    }


    //draw picture
    //Mat image_match_r;
    //drawMatches(src_image_l, leftInlier2, temp_image_r, rightInlier2, inlierMatch2_sorted, image_match_r);
    //imwrite("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/featureImage/matched_src_l_temp_r.png",image_match_r);


    for(int i = 0;i<int(key_points_src_l_temp_l.size());++i)
    {
        if(features_in_image[i].m_distance_from_edge[0]>0 && features_in_image[i].m_distance_from_edge[1]>0 )
        {           

            double min_y_distance = 9999.0;
            int min_index;
            for(int j = int(key_points_src_l_temp_l.size());j<features_in_image.size();++j)
            {
                if(features_in_image[j].m_distance_from_edge[0]<0 || features_in_image[j].m_distance_from_edge[1]<0 )
                    continue;

                double current_distance = abs(features_in_image[i].m_distance_from_edge[1] - features_in_image[j].m_distance_from_edge[1]);
                if(current_distance < min_y_distance)
                {
                    min_y_distance = current_distance;
                    min_index = j;
                }
            }

            //no larger than 0.1mm
            if(min_y_distance < DISTANCE_TRESHOLD)
            {
                cout<<"min_y_distance is :"<<min_y_distance<<endl;
                features_selected.emplace_back(features_in_image[i]);
                features_selected.emplace_back(features_in_image[min_index]);
            }

            /*
            for(int j = int(key_points_src_l_temp_l.size());j<features_in_image.size();++j)
            {               
                if(features_in_image[j].m_distance_from_edge[0]<0 || features_in_image[j].m_distance_from_edge[1]<0 )
                    continue;

                cout<<"test selected point"<<endl;

                features_selected.emplace_back(features_in_image[i]);
                features_selected.emplace_back(features_in_image[j]);
            }
            */
        }

    }
}

void Common::SavePLYFile_PointCloud(std::string filePath, std::vector<Structure> structure)
{
    ofstream ofs(filePath);
    if (!ofs)
    {
        cout << "err in create ply!" << endl;;
        return;
    }
    else
    {
        cout << "BEGIN SAVE PLY" << endl;
        ofs << "ply " << endl << "format ascii 1.0" << endl;
        //ofs << "element vertex " << this->structures.size() + this->Views.size() << endl;//old
        ofs << "element vertex " << structure.size()  << endl;
        ofs << "property float x" << endl << "property float y" << endl << "property float z" << endl;
        ofs << "property uchar blue"
            << endl << "property uchar green"
            << endl << "property uchar red" << endl;
        ofs << "end_header" << endl;


        for (uint i = 0; i < structure.size(); i++)
        {
            ofs << structure[i].positions_array[0] << " " << structure[i].positions_array[1] << " " << structure[i].positions_array[2]
                << " 255 255 255" << endl;
        }
    }
    ofs.close();
    ofs.flush();
    cout << "FINISH SAVE PLY" << endl;
}

void Common::CalculateStructure_Init_DLT(std::vector<View> view, std::vector<Observation> obs , vector<Structure> & P )
{
    cout << "START INIT POINTCLOUD" << endl;
    //i < 2 -> i < 1
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < obs.size(); j++)
        {
            Eigen::Matrix<double, 3, 4> P1, P2;

            int host_camera_id = (i == 0 ? obs[j].host_camera_id : obs[j].neighbor_camera_id);
            int neighbor_camera_id = (i == 0 ? obs[j].neighbor_camera_id : obs[j].host_camera_id);

            P1.block(0, 0, 3, 3) = view[host_camera_id].rotation;
            P1.block(0, 3, 3, 1) = -view[host_camera_id].t;
            P1 = view[host_camera_id].K * P1;

            P2.block(0, 0, 3, 3) = view[neighbor_camera_id].rotation;
            P2.block(0, 3, 3, 1) = -view[neighbor_camera_id].t;
            P2 = view[neighbor_camera_id].K * P2;

            Eigen::Vector2d pixel = (i == 0 ? obs[j].pixel : obs[j].match_pixel);
            Eigen::Vector2d match_pixel = (i == 0 ? obs[j].match_pixel : obs[j].pixel);

            Eigen::Vector3d X_init;
            Eigen::Matrix<double, 4, 4> A;
            A.block(0, 0, 1, 4) =
                (pixel.x()) * P1.row(2) - P1.row(0);
            A.block(1, 0, 1, 4) =
                (pixel.y()) * P1.row(2) - P1.row(1);
            A.block(2, 0, 1, 4) =
                (match_pixel.x()) * P2.row(2) - P2.row(0);
            A.block(3, 0, 1, 4) =
                (match_pixel.y()) * P2.row(2) - P2.row(1);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
            Eigen::Vector4d X_ = svd.matrixV().col(3);
            X_init << X_.x() / X_.w(), X_.y() / X_.w(), X_.z() / X_.w();
            //cout<<"X_init.z(): "<<X_init.z()<<endl;
            P.emplace_back(Structure(X_init, i*obs.size() + j, true));

        }
    }
    cout << "FINISH INIT POINTCLOUD" << endl;
}


void Common::CalculateStructure_Ceres(std::vector<View> view, std::vector<Observation> obs, vector<Structure> & output_struct)
{

    cout << "SATRT  BA" << endl;
    ceres::Problem problem;
    ceres::LossFunction* lossFunc = new ceres::HuberLoss(2.0f);

    for (uint i = 0; i < output_struct.size(); ++i)
    {
        if (!output_struct[i].isvalid_structure)
            continue;

        int current_index = i / obs.size();     
        Eigen::Matrix<double, 3, 4> P;

        P.block(0, 0, 3, 3) = view[current_index].rotation;
        P.block(0, 3, 3, 1) = view[current_index].t;
        P = view[current_index].K * P;
        Eigen::Vector2d pixel = (current_index == 0 ? obs[i].pixel : obs[i - obs.size()].match_pixel);
        /*
        ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<Ceres_Triangulate, 2, 3>(new Ceres_Triangulate(pixel, P));
        problem.AddResidualBlock(cost_function, lossFunc, output_struct[i].positions_array);
        */

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<Ceres_Triangulate_AdjustRt, 2, 3, 3, 3>(new Ceres_Triangulate_AdjustRt(pixel, view[current_index].K ));
        problem.AddResidualBlock(cost_function, lossFunc, output_struct[i].positions_array, view[current_index].rotation_array, view[current_index].translatrion_array);


    }

    ceres::Solver::Options options;
    options.max_num_iterations = 20;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    cout << "FINISH CERES BA" << endl;

}

void Common::CalculateDistance(vector<FeatureInImage> features_selected, vector<Structure> P, double radius)
{
    int temp_l_size = int(features_selected.size())/2 ;
    assert(temp_l_size != 0);
    double distance = 0.0;
    vector<double> distance_vec;
    for(int i = 0;i<temp_l_size;++i)
    {
        double s_pow = 0.0;
        s_pow = pow(abs(P[2 * i].position.x() - P[2 * i + 1].position.x()),2) + pow(abs(P[2 * i].position.y() - P[2 * i + 1].position.y()),2) + pow(abs(P[2 * i].position.z() - P[2 * i + 1].position.z()),2);
        double alpha1 = features_selected[2 * i].m_distance_from_edge[1]/radius;
        double alpha2 = features_selected[2 * i + 1].m_distance_from_edge[1]/radius;
        double abs_alpha = abs(alpha1 - alpha2);
        //cout<<"abs_alpha: "<<abs_alpha<<endl;
        double local_distance = sqrt(s_pow - 2 * radius * radius *(1 - cos(abs_alpha)));
        //cout<<"s: "<<sqrt(s_pow)<<endl;
        //cout<<"local_distance: "<<local_distance<<endl;
        local_distance += features_selected[2 * i].m_distance_from_edge[0] + features_selected[2 * i + 1].m_distance_from_edge[0];
        //cout<<"features_selected[2 * i].m_distance_from_edge[0] + features_selected[2 * i + 1].m_distance_from_edge[0]: "<<features_selected[2 * i].m_distance_from_edge[0] + features_selected[2 * i + 1].m_distance_from_edge[0]<<endl;
        //cout<< "distance: "<<local_distance<<endl;
        distance_vec.push_back(local_distance);
        //distance = distance + local_distance;

    }

    double sum = std::accumulate(std::begin(distance_vec), std::end(distance_vec), 0.0);
    double mean = sum / distance_vec.size(); //均值

    double accum = 0.0;
    std::for_each(std::begin(distance_vec), std::end(distance_vec), [&](const double d) {
        accum += (d - mean)*(d - mean);
    });

    double stdev = sqrt(accum / (distance_vec.size() - 1)); //方差

    cout << "stdev : " << stdev << endl;
    //去除 std +- 1 * stdev 外的数据，再算一遍均值
    vector<double> distance_vec_fliter;
    for (auto dv : distance_vec)
    {
        if (dv <= (mean + 1 * stdev) && dv > (mean - 1 * stdev))
        {
            distance_vec_fliter.push_back(dv);            
        }
    }
    double sum_fliter = std::accumulate(std::begin(distance_vec_fliter), std::end(distance_vec_fliter), 0.0);
    double mean_filter = sum_fliter / distance_vec_fliter.size(); //均值

    cout << "final diatance: " << mean_filter << " mm" << endl;

}

void Common::DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2d>points1, vector<Point2d>points2, cv::Mat F,string save_path)
{
    cout<<"points1 size:"<<points1.size()<<endl;
    cout<<"F: "<<F<<endl;
    std::vector<cv::Vec<double, 3>> epilines;
    cv::computeCorrespondEpilines(points1, 1, F, epilines);//ŒÆËã¶ÔÓŠµãµÄÍâŒ«ÏßepilinesÊÇÒ»žöÈýÔª×é(a,b,c)£¬±íÊŸµãÔÚÁíÒ»ÊÓÍŒÖÐ¶ÔÓŠµÄÍâŒ«Ïßax+by+c=0;
                                                           //œ«ÍŒÆ¬×ª»»ÎªRGBÍŒ£¬»­ÍŒµÄÊ±ºòÍâŒ«ÏßÓÃ²ÊÉ«»æÖÆ
    cv::Mat img1, img2;
    if (img_1.type() == CV_8UC3)
    {
        img_1.copyTo(img1);
        img_2.copyTo(img2);
    }
    else if (img_1.type() == CV_8UC1)
    {
        cvtColor(img_1, img1, COLOR_GRAY2BGR);
        cvtColor(img_2, img2, COLOR_GRAY2BGR);
    }
    else
    {
        cout << "unknow img type\n" << endl;
        exit(0);
    }

    cv::RNG& rng = theRNG();
    for (int i = 0; i < points1.size(); i += 1)
    {
        Scalar color = Scalar(rng(256), rng(256), rng(256));//Ëæ»ú²úÉúÑÕÉ«
        circle(img1, points1[i], 2, color, 2);//ÔÚÊÓÍŒ1ÖÐ°Ñ¹ØŒüµãÓÃÔ²ÈŠ»­³öÀŽ
        circle(img2, points2[i], 2, color, 2);//ÔÚÊÓÍŒ2ÖÐ°Ñ¹ØŒüµãÓÃÔ²ÈŠ»­³öÀŽ£¬È»ºóÔÙ»æÖÆÔÚ¶ÔÓŠµãŽŠµÄÍâŒ«Ïß
        line(img2, Point(0, -epilines[i][2] / epilines[i][1]), Point(img2.cols, -(epilines[i][2] + epilines[i][0] * img2.cols) / epilines[i][1]), color);
    }
    string img_pointLines = save_path + "/img_pointLines.png";
    string img_point = save_path + "/img_point.png";
    imwrite(img_point, img1);
    imwrite(img_pointLines, img2);
}

void Common::Cylinderfitting(vector<Structure> P, double & ridius)
{
    cout<<"start fitting Cylinder..."<<endl;
    int i=0,j=0;
     int int_temp=0;
     int loop_times=0,pp=0;
     double a=1,b=1,c=1;
     double x0=0,y0=0,z0=0;
     double D=0,s=0,S=0,dx=0,dy=0,dz=0;
     double R=0;
     double d_temp1=0,d_temp2=0,d_temp3=0;
     double B[MAX_POINT_NUM][7]={0};
     double L[MAX_POINT_NUM]   ={0};
     double worldVetex[MAX_POINT_NUM][3]={0};

     double mean_x=0,mean_y=0,mean_z=0;
     bool while_flag=1;
     CvMat* C    = cvCreateMat( 2, 7, CV_64FC1 );
     CvMat* W    = cvCreateMat( 2, 1, CV_64FC1 );
     CvMat* N    = cvCreateMat( 9, 9, CV_64FC1 );
     CvMat* N_inv= cvCreateMat( 9, 9, CV_64FC1 );
     CvMat* UU   = cvCreateMat( 9, 1, CV_64FC1 );
     CvMat* para = cvCreateMat( 9, 1, CV_64FC1 );

     cvZero(C);cvZero(W);cvZero(N);cvZero(N_inv);cvZero(UU);
     cvSetIdentity(para);

     //test the code
/*
     ifstream Points_in("/home/jiajie/3d_reco/Binocular/BinocularCalibration1.1/points.txt");
        int_temp=0;
        if (Points_in.is_open())
        {
          while (!Points_in.eof())
          {
             Points_in>>worldVetex[int_temp][0]>>worldVetex[int_temp][1]>>worldVetex[int_temp][2];
             int_temp++;
          }
        }
        else
        {
             cout<<"open fail!"<<endl;
        }
        int_temp=int_temp-1;
*/

     for(int i = 0;i<P.size();++i)
     {
         worldVetex[i][0] = P[i].position.x();
         worldVetex[i][1] = P[i].position.y();
         worldVetex[i][2] = P[i].position.z();
     }

     int_temp = int(P.size())-1;


     for(i=0;i<int_temp;i++)
     {
        d_temp1+=worldVetex[i][0];
        d_temp2+=worldVetex[i][1];
        d_temp3+=worldVetex[i][2];
     }
     mean_x=d_temp1/int_temp; mean_y=d_temp2/int_temp;mean_z=d_temp3/int_temp;
     x0=mean_x;y0=mean_y;z0=mean_z;

     double min_temp_change = 1e-7;
     double current_temp_change = 10000;
     double last_temp = 0;

  while(while_flag==true && current_temp_change > min_temp_change)
  {   
      if(a<0)
      {
         a=-a;b=-b;c=-c;
      }
      if(a==0)
      {
         if(b<0)
         {
           b=-b;c=-c;
         }
         if(b==0)
         {
           if(c<0)
             c=-c;
         }
      }
       s=sqrt(pow(a,2)+pow(b,2)+pow(c,2))+0.0000001;
       a=a/s+0.0000001;b=b/s+0.0000001;c=c/s+0.0000001;

      for(i=0;i<int_temp;i++)
      {

        D=a*(worldVetex[i][0]-x0)+b*(worldVetex[i][1]-y0)+c*(worldVetex[i][2]-z0);       //D=a*(X(i)-x0)+b*(Y(i)-y0)+c*(Z(i)-z0);
        dx=x0+a*D-worldVetex[i][0];dy=y0+b*D-worldVetex[i][1];dz=z0+c*D-worldVetex[i][2];//dx=x0+a*D-X(i);dy=y0+b*D-Y(i);dz=z0+c*D-Z(i);
        S=sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));                                           //S=sqrt(dx^2+dy^2+dz^2);
        B[i][0]=(dx*(a*(worldVetex[i][0]-x0)+D)+dy*b*(worldVetex[i][0]-x0)+dz*c*(worldVetex[i][0]-x0))/S;   //b1
        B[i][1]=(dx*a*(worldVetex[i][1]-y0)+dy*(b*(worldVetex[i][1]-y0)+D)+dz*c*(worldVetex[i][1]-y0))/S;
        B[i][2]=(dx*a*(worldVetex[i][2]-z0)+dy*b*(worldVetex[i][2]-z0)+dz*(c*(worldVetex[i][2]-z0)+D))/S;
        B[i][3]=(dx*(1-pow(a,2))-dy*a*b-dz*a*c)/S;
        B[i][4]=(-dx*a*b+dy*(1-pow(b,2))-dz*b*c)/S;
        B[i][5]=(-dx*a*c-dy*b*c+dz*(1-pow(c,2)))/S;
        B[i][6]=-1;


                   //B=[B;b1 b2 b3 b4 b5 b6 b7];
        L[i]=R-S;  //l=[R-S];
                   //L=[L;l];
      }

      d_temp1=1-pow(a,2)-pow(b,2)-pow(c,2);

      if(fabs(a)>=fabs(b) && fabs(a)>=fabs(c))
      {
        cvZero(C);cvZero(W);
        cvmSet( C, 0, 0, 2*a );cvmSet( C, 0, 1, 2*b );
        cvmSet( C, 0, 2, 2*c );cvmSet( C, 1, 3, 1 );
        cvmSet( W, 0, 0, d_temp1 );cvmSet( W, 1, 0, mean_x-x0 );
      }
      if(fabs(b)>=fabs(a) && fabs(b)>=fabs(c))
      {
        cvZero(C);cvZero(W);
        cvmSet( C, 0, 0, 2*a );cvmSet( C, 0, 1, 2*b );
        cvmSet( C, 0, 2, 2*c );cvmSet( C, 1, 4, 1 );
        cvmSet( W, 0, 0, d_temp1 );cvmSet( W, 1, 0, mean_y-y0 );
      }
      if(fabs(c)>=fabs(a) && fabs(c)>=fabs(b))
      {
        cvZero(C);cvZero(W);
        cvmSet( C, 0, 0, 2*a );cvmSet( C, 0, 1, 2*b );
        cvmSet( C, 0, 2, 2*c );cvmSet( C, 1, 5, 1 );
        cvmSet( W, 0, 0, d_temp1 );cvmSet( W, 1, 0, mean_z-z0 );
      }

        //Nbb=B'*B;U=B'*L;
        //N=[Nbb C';C zeros(2)];
        //UU=[U;W];
        //para=inv(N)*UU;

      cvZero(N);        // N= |Nbb(7*7)  C'(7*2)|
                        //    |C  (2*7)  O(2*2) |
      for(i=0;i<7;i++)
      {
         for(j=0;j<7;j++)
         {
            d_temp1=0;
            for(pp=0;pp<int_temp;pp++)
            {
              d_temp1+=B[pp][i]*B[pp][j];
            }
            cvmSet(N,i,j,d_temp1);
         }
      }
      for(i=0;i<2;i++)
         for(j=0;j<7;j++)
            cvmSet(N,i+7,j,cvmGet(C,i,j));
      for(i=0;i<2;i++)
         for(j=0;j<7;j++)
            cvmSet(N,j,i+7,cvmGet(C,i,j));

      for(i=0;i<7;i++)
      {
         d_temp1=0;
         for(pp=0;pp<int_temp;pp++)
         {
           d_temp1+=B[pp][i]*L[pp];
         }
         cvmSet(UU,i,0,d_temp1);
      }
      for(i=0;i<2;i++)
         cvmSet(UU,i+7,0,cvmGet(W,i,0));

      cvInvert(N,N_inv);           //para=inv(N)*UU;
      cvMatMul(N_inv,UU,para);
      a = a+cvmGet(para,0,0);      //a=a+para(1);b=b+para(2);c=c+para(3);
      b = b+cvmGet(para,1,0);
      c = c+cvmGet(para,2,0);
      x0= x0 + cvmGet(para,3,0);   //x0=x0+para(4);y0=y0+para(5);z0=z0+para(6);
      y0= y0 + cvmGet(para,4,0);
      z0= z0 + cvmGet(para,5,0);
      R = R  + cvmGet(para,6,0);
      loop_times=loop_times+1;     //t=t+1

      d_temp1=cvmGet(para,0,0);
      for(i=1;i<7;i++)
      {
        if(fabs(d_temp1)<fabs(cvmGet(para,i,0)))
           d_temp1=cvmGet(para,i,0);
      }
      current_temp_change = fabs(last_temp - fabs(d_temp1));
      last_temp = fabs(d_temp1);
      cout<<"fabs(d_temp1): "<<fabs(d_temp1)<<endl;
      if(fabs(d_temp1)>0.0000001)
        while_flag=1;
      else
        while_flag=0;
  }
    cout<<"loop_times is: "<<loop_times<<endl;
    cout<<"derection vector:["<<a<<", "<<b<<", "<<c<<"]"<<endl;
    cout<<"points on axis:["<<x0<<", "<<y0<<", "<<z0<<"]"<<endl;
    cout<<"R : "<<R<<endl;

    cout << "SATRT  BA" << endl;
    ceres::Problem problem;
    ceres::LossFunction* lossFunc = new ceres::HuberLoss(2.0f);

    //init paras of Cylinder a,b,c,x0,y0,z0,R
    double *para_array = new double[1];
    para_array[0] = R;

    vector<double> para_array_fixed;
    para_array_fixed.push_back(a);
    para_array_fixed.push_back(b);
    para_array_fixed.push_back(c);
    para_array_fixed.push_back(x0);
    para_array_fixed.push_back(y0);
    para_array_fixed.push_back(z0);

    for (uint i = 0; i < P.size(); ++i)
    {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<Ceres_refine_radius, 1, 1>(new Ceres_refine_radius(P[i].position, para_array_fixed));
        problem.AddResidualBlock(cost_function, lossFunc, para_array);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 30;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    cout<<"BA derection vector:["<<para_array_fixed[0]<<", "<<para_array_fixed[1]<<", "<<para_array_fixed[2]<<"]"<<endl;
    cout<<"BA points on axis:["<<para_array_fixed[3]<<", "<<para_array_fixed[4]<<", "<<para_array_fixed[5]<<"]"<<endl;
    cout<<"BA R : "<<para_array[0]<<endl;
    ridius = para_array[0];

    cout << "FINISH CERES BA" << endl;
}

void Common::PclCaculateRadius(vector<Structure> P, double & radius, Config cfg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);

    for(int i = 0;i<P.size();++i)
    {
        pcl::PointXYZ point;
        point.x = 1 * P[i].position.x();
        point.y = 1 * P[i].position.y();
        point.z = 1 * P[i].position.z();
        pointCloud->points.push_back(point);
    }

    // Define random generator with Gaussian distribution  test
    /*
    const double mean = 0.0;//均值
    const double stddev = 0.2;//标准差
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);


    //生成圆柱点云

    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud1(new pcl::PointCloud<pcl::PointXYZ>);

    for (float z(-1); z <= 1; z += 0.2)
      {
          for (float angle(0.0); angle <= 110.0; angle += 3.0)
          {
              pcl::PointXYZ basic_point;
              basic_point.x = 136 + 2.5*cos(angle / 180 * M_PI) + dist(generator);
              basic_point.y = -9 + 2.5*sin(angle / 180 * M_PI) + dist(generator);
              basic_point.z = -298 + z + dist(generator) ;
              pointCloud1->points.push_back(basic_point);
          }
      }
      */

    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_fit(new pcl::PointCloud<pcl::PointXYZ>);
    fitCylinder(pointCloud, pointCloud_fit, radius, cfg);
    if(cfg.m_visualization)
    {
        boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer(new pcl::visualization::PCLVisualizer("Ransac"));
        viewer->setBackgroundColor(0, 0, 0);
        //创建窗口
        int vp;
        viewer->createViewPort(0.0, 0.0, 1.0, 1.0, vp);
        //设置点云颜色
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(pointCloud, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(pointCloud_fit, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(pointCloud, source_color, "source", vp);
        viewer->addPointCloud<pcl::PointXYZ>(pointCloud_fit, target_color, "target", vp);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");
        viewer->spin();
    }

}

// use ransanc to fit cylinder
void fitCylinder(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out, double &radius, Config cfg)
{
    // Create segmentation object for cylinder segmentation and set all the parameters
    pcl::ModelCoefficients::Ptr coeffients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);

    //  Normals

    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ>>(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normalsFilter(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimator;
    normalEstimator.setSearchMethod(tree);
    normalEstimator.setInputCloud(cloud_in);
    normalEstimator.setKSearch(50);
    normalEstimator.compute(*normalsFilter);


    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);

    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(cfg.m_pcl_fit_distance);
    //seg.setProbability(0.99);
    seg.setRadiusLimits(0, 100);
    seg.setInputCloud(cloud_in);
    seg.setInputNormals(normalsFilter);

    // Perform segment
    seg.segment(*inliers_cylinder, *coeffients_cylinder);

    // Ouput extracted cylinder
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers_cylinder);
    extract.filter(*cloud_out);

    pcl::PointXYZ point;
    point.x = coeffients_cylinder->values[0];
    point.y = coeffients_cylinder->values[1];
    point.z = coeffients_cylinder->values[2];
    cloud_out->points.push_back(point);

    float sum_D = 0.0;
    float sum_Ave = 0.0;
    float x0 = coeffients_cylinder->values[0];
    float y0 = coeffients_cylinder->values[1];
    float z0 = coeffients_cylinder->values[2];
    float l = coeffients_cylinder->values[3];
    float m= coeffients_cylinder->values[4];
    float n = coeffients_cylinder->values[5];
    float r0 = coeffients_cylinder->values[6];  
    radius = r0;
    for (int i = 0; i < cloud_out->points.size(); i++) {
        float x = cloud_out->points[i].x;
        float y = cloud_out->points[i].y;
        float z = cloud_out->points[i].z;
        // D=part1+part2
        float part1 = pow(x - x0, 2) + pow(y - y0, 2) + pow(z - z0, 2) - pow(r0, 2);
        float part2 = -pow(l*(x - x0) + m * (y - y0) + n * (z - z0), 2) / (l*l + m * m + n * n);
        sum_D +=pow( part1 + part2,2);
        sum_Ave += fabs(part1 + part2);
    }
}

Eigen::MatrixXd toEigenMatrixXd(const cv::Mat &cvMat)
{
    Eigen::MatrixXd eigenMat;
    eigenMat.resize(cvMat.rows, cvMat.cols);
    for (int i=0; i<cvMat.rows; i++)
        for (int j=0; j<cvMat.cols; j++)
            eigenMat(i,j) = cvMat.at<double>(i,j);

    return eigenMat;
}

vector<double> toStdVector(const cv::Mat &cvMat)
{
    vector<double> stdVec;
    for (int i=0; i<cvMat.rows; i++)
        for (int j=0; j<cvMat.cols; j++)
            stdVec.emplace_back(cvMat.at<double>(i,j));

    return stdVec;

}

//最大最小聚类
void MaxMinCluster(vector<cv::Point2d> data,vector<vector<cv::Point2d>> & result, Config cfg)
{
  assert(data.size()>0);
  //adative?
  double t = cfg.m_cluster_threshold;
  vector<cv::Point2d> cluster_center; //聚类中心集，选取第一个模式样本作为第一个聚类中心Z1
  cluster_center.emplace_back(data[0]);
  //第2步：寻找Z2,并计算阈值T
  double T = step2(data, t, cluster_center);
  // 第3,4,5步，寻找所有的聚类中心
  get_clusters(data, cluster_center, T);
  // 按最近邻分类
  result = classify(data, cluster_center, T);
}

bool sort_by_vec_x(Point2d p1 ,Point2d p2)
{
    return p1.x<p2.x;
}

bool sort_by_vec_y(Point2d p1 ,Point2d p2)
{
    return p1.y<p2.y;
}

bool sort_by_vec_size(vector<Point2d> p1 ,vector<Point2d> p2)
{
    return p1.size()>p2.size();
}

// 计算两个模式样本之间的欧式距离
//仅考虑二维点，模板还不会用...
double get_distance(cv::Point2d data1, cv::Point2d data2)
{
  double distance = 0.f;
  distance = pow((data1.x-data2.x), 2) + pow((data1.y-data2.y), 2);
  return sqrt(distance);
}

// 寻找Z2,并计算阈值T
double step2(vector<cv::Point2d> data,double t, vector<cv::Point2d> & cluster_center)
{
  double distance = 0.f;
  int index = 0;
  for(uint i = 0;i<data.size();++i)
  {
    //欧氏距离
    double temp_distance = get_distance(data[i], cluster_center[0]);
    if(temp_distance>distance)
    {
      distance = temp_distance;
      index = i;
    }
  }
  //将Z2加入到聚类中心集中
  cluster_center.emplace_back(data[index]);
  double T = t * distance; //距离阈值，maxmin聚类的终止条件是： 最大最小距离不大于该阈值
  return T;
}

void get_clusters(vector<cv::Point2d> data,vector<cv::Point2d> & cluster_center, double T)
{
  double max_min_distance = 0.f;
  int index = 0;
  for(uint i = 0;i<data.size();++i)
  {
    vector<double> min_distance_vec;
    for(uint j = 0;j<cluster_center.size();++j)
    {
      double distance = get_distance(data[i],cluster_center[j]);
      min_distance_vec.emplace_back(distance);
    }
    double min_distance = *min_element(min_distance_vec.begin(),min_distance_vec.end());
    if(min_distance>max_min_distance)
    {
      max_min_distance = min_distance;
      index = i;
    }

  }

  if(max_min_distance > T)
  {
    cluster_center.emplace_back(data[index]);
    //迭代
    get_clusters(data, cluster_center, T);
  }

}

//最近邻分类（离哪个聚点中心最近，归为哪类）
vector<vector<cv::Point2d>> classify(vector<cv::Point2d> data, vector<cv::Point2d> & cluster_center, double T)
{
  int vec_size = cluster_center.size();
  vector<vector<cv::Point2d>> result(vec_size);
  for(uint i = 0;i<data.size();++i)
  {
    double min_distance = T;
    int index = 0;
    //cout<<"cluster_center size: "<<cluster_center.size()<<endl;
    vector<cv::Point2d> temp_vec;
    for(uint j = 0;j<cluster_center.size();++j)
    {
      //cout<<"j is: "<<j<<endl;
      double temp_distance = get_distance(data[i], cluster_center[j]);
      if(temp_distance < min_distance)
      {
        min_distance = temp_distance;
        index = j;
        //cout<<"index is: "<<index<<endl;
      }
    }
    result[index].emplace_back(data[i]);

  }
  return result;
}




