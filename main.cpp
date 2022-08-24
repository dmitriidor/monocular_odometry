#include <iostream>
#include <ctime>
#include <numeric>

#include <eigen3/Eigen/Dense>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
// #include <opencv4/opencv2/core/eigen.hpp>

#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/xfeatures2d/cuda.hpp>

#include "pose_estimator/pose_estimator_class.hpp"
#include "angle_converter/angle_converter_class.hpp"
#include "tile_depthmeter/tile_depthmeter_class.hpp"

using namespace std;
using namespace cv;
using namespace Eigen; 

int main(int, char**) {


    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_3.avi");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }


    Mat frame_1, frame_2; 
    cap.set(CAP_PROP_POS_MSEC, 6000);
    cap >> frame_1;

    cuda::GpuMat gpu_frame_1;
    cuda::GpuMat gpu_frame_2; 
    cuda::GpuMat gpu_frame_filtered;

    gpu_frame_2.upload(frame_1);
    cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_16U, CV_16U, Size(7, 7), 1, 1);
    Ptr<cuda::Filter> laplace_filter = cuda::createLaplacianFilter(CV_16U, CV_16U, 1); 
    // cuda::SURF_CUDA surf(1500, 4, 2, true);
    Ptr<cuda::ORB> feature_extractor = cuda::ORB::create(100, 1.2f, 8, 40, 0, 2, 0, 50);

    vector<KeyPoint> keys_1, keys_2;
    cuda::GpuMat gpu_keys_1, gpu_keys_2;

    // vector<float> descriptors_1, descriptors_2; 
    cuda::GpuMat gpu_descriptors_1, gpu_descriptors_2; 

    gpu_frame_1 = gpu_frame_2; 
    // surf(gpu_frame_2, cuda::GpuMat(), gpu_keys_2, gpu_descriptors_2);
    feature_extractor->detectAndCompute(gpu_frame_2, noArray(), keys_2, gpu_descriptors_2);

    // Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);

    clock_t start, end; 
    for(int iter = 1; iter < 1000000+1; iter++) {
        cap >> frame_2;

        // gpu_keys_1 = gpu_keys_2; 
        keys_1 = keys_2; 
        gpu_descriptors_1 = gpu_descriptors_2;
        gpu_frame_1 = gpu_frame_2; 
        

        gpu_frame_2.upload(frame_2); 
        gpu_frame_filtered.upload(frame_2);
        cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);
        cuda::cvtColor(gpu_frame_filtered, gpu_frame_filtered, COLOR_BGR2GRAY);
        gpu_frame_filtered.convertTo(gpu_frame_filtered, CV_16U);
        cuda::normalize(gpu_frame_filtered, gpu_frame_filtered, 0, pow(2, 16), NORM_MINMAX, CV_16U);
        gaussian_filter->apply(gpu_frame_filtered, gpu_frame_filtered);
        laplace_filter->apply(gpu_frame_filtered, gpu_frame_filtered); 
        double minimum_value, maximum_value; 
        cuda::minMax(gpu_frame_filtered, &minimum_value, &maximum_value);  
        gpu_frame_2.convertTo(gpu_frame_2, CV_16U);
        cuda::normalize(gpu_frame_2, gpu_frame_2, 0, pow(2, 16), NORM_MINMAX, CV_16U);
        cuda::subtract(gpu_frame_2, gpu_frame_filtered, gpu_frame_2);
        cuda::minMax(gpu_frame_2, &minimum_value, &maximum_value); 
        gpu_frame_2.convertTo(gpu_frame_2, CV_8U, 1/pow(2, 8));

        // surf(gpu_frame_2, cuda::GpuMat(), gpu_keys_2, gpu_descriptors_2);
        feature_extractor->detectAndCompute(gpu_frame_2, noArray(), keys_2, gpu_descriptors_2);

        vector< vector<DMatch> > matches; 
        matcher->knnMatch(gpu_descriptors_1, gpu_descriptors_2, matches, 2);

        vector<DMatch> filtered_matches; 
        double ratio = 0.1; 
        for(int i = 0; i < matches.size(); i++) {
            if(matches[i][0].distance < matches[i][1].distance*ratio) {
                filtered_matches.push_back(matches[i][0]);
            }
        }

        // surf.downloadKeypoints(gpu_keys_1, keys_1);
        // surf.downloadKeypoints(gpu_keys_2, keys_2);

        vector<Point2f> filtered_points_1(filtered_matches.size(), Point2f(0, 0));
        vector<Point2f> filtered_points_2(filtered_matches.size(), Point2f(0, 0)); 
        for(int i = 0; i < filtered_matches.size(); i++) {
            filtered_points_1[i] = keys_1[filtered_matches[i].queryIdx].pt;
            filtered_points_2[i] = keys_2[filtered_matches[i].trainIdx].pt;
        }

        double sum_x, sum_y; 

        sum_x = 0; 
        sum_y = 0;
        for(int i = 0; i < filtered_points_1.size(); i++){
            sum_x += filtered_points_1[i].x;
            sum_y += filtered_points_1[i].y;
        }
        Point2f centroid_1 =  Point2f(sum_x/filtered_points_1.size(), sum_y/filtered_points_1.size());


        sum_x = 0; 
        sum_y = 0;
        for(int i = 0; i < filtered_points_2.size(); i++){
            sum_x += filtered_points_2[i].x;
            sum_y += filtered_points_2[i].y;
        }
        Point2f centroid_2 =  Point2f(sum_x/filtered_points_2.size(), sum_y/filtered_points_2.size());

        for(int i = 0; i < filtered_points_1.size(); i++){
            filtered_points_1[i].x -= centroid_1.x;
            filtered_points_1[i].y -= centroid_1.y;
        }


        for(int i = 0; i < filtered_points_2.size(); i++){
            filtered_points_2[i].x -= centroid_2.x;
            filtered_points_2[i].y -= centroid_2.y;
        }

        Mat points3d_1 = Mat::zeros(Size(filtered_points_1.size(), 3), CV_32F); 
        Mat points3d_2 = Mat::zeros(Size(filtered_points_1.size(), 3), CV_32F); 

        for(int i = 0; i < points3d_1.cols; i++){
            points3d_1.at<float>(0, i) = filtered_points_1[i].x;
            points3d_1.at<float>(1, i) = filtered_points_1[i].y;
            points3d_1.at<float>(2, i) = 1; 
        }

        for(int i = 0; i < points3d_2.cols; i++){
            points3d_2.at<float>(0, i) = filtered_points_2[i].x;
            points3d_2.at<float>(1, i) = filtered_points_2[i].y;
            points3d_2.at<float>(2, i) = 1; 
        }

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pnts3d_1_eigen(points3d_1.ptr<float>(), points3d_1.rows, points3d_1.cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pnts3d_2_eigen(points3d_2.ptr<float>(), points3d_2.rows, points3d_2.cols);

        if(iter < 4)
            continue;
            
        MatrixXf H = pnts3d_1_eigen * pnts3d_2_eigen.transpose(); 

        cout << H << endl;

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullV | Eigen::ComputeFullU);

        Matrix3f R = svd.matrixV() * svd.matrixU().transpose(); 

        cout << R << endl;

        if(R.determinant() < 0) { 
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
            MatrixXf Vt = svd.matrixV(); 
            Eigen::Vector3f M = Eigen::Vector3f(1, 1, -1);
            Eigen::Matrix3Xf C = Vt.array().colwise() * M.array();
            R =  Vt * svd.matrixU().transpose();
        }

        Vector3f ang = R.eulerAngles(2, 1, 0); 
        cout << ang*360/3.14 << endl; 

        Vector3f centroid_1_eigen;
        centroid_1_eigen << centroid_1.x, centroid_1.y, 1; 
        Vector3f centroid_2_eigen;
        centroid_2_eigen << centroid_2.x, centroid_2.y, 1; 

        Vector3f T = centroid_2_eigen - centroid_1_eigen;

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);

        Mat filtered_matches_img; 
        drawMatches(frame_1, keys_1, frame_2, keys_2, filtered_matches, filtered_matches_img);

        Mat shit; 
        gpu_frame_filtered.download(shit); 
    
        imshow("Frame 1", frame_1);
        imshow("Frame 2", frame_2);
        // imshow("TD: ", depthmeter.pipeline.frame_laplacian);
        // imshow("Matches", matches_img);
        imshow("Filtered Matches", filtered_matches_img);

        waitKey(0);
    }
};

/*
node: 
1. get image, get delta
2. do filtering
3. extract features 
4. call visual odometery 
>> 1. calcualate essential matrix
>> 2. filter RANSAC
>> 3. calculate speed
5. form message 
*/
