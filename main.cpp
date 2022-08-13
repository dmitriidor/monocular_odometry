#include <iostream>
#include <ctime>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include "pose_estimator/pose_estimator_class.hpp"
#include "angle_converter/angle_converter_class.hpp"
#include "tile_depthmeter/tile_depthmeter_class.hpp"

using namespace std;
using namespace cv;

int main(int, char**) {


    VideoCapture cap("test_vids/circular.avi");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // cap.set(CAP_PROP_POS_MSEC, 10);
    // cap >> frame_1;
    // cap.set(CAP_PROP_POS_MSEC, 100);
    // cap >> frame_2;
    frame_1 = imread("test_vids/pools_closed.jpg", IMREAD_COLOR);
    frame_2 = imread("test_vids/pools_closed.jpg", IMREAD_COLOR);

    cuda::GpuMat gpu_frame_1;
    cuda::GpuMat gpu_frame_2; 
    gpu_frame_1.upload(frame_1);
    gpu_frame_2.upload(frame_2); 

    cuda::cvtColor(gpu_frame_1, gpu_frame_1, COLOR_BGR2GRAY);
    cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(gpu_frame_1.type(), gpu_frame_1.type(), Size(5, 5), 0, 0);

    gaussian_filter->apply(gpu_frame_1, gpu_frame_1);
    gaussian_filter->apply(gpu_frame_2, gpu_frame_2);

    // Ptr<cuda::Filter> sobel_filter = cuda::createSobelFilter(gpu_frame_1.type(), gpu_frame_1.type(), 2, 2, 7); 

    Ptr<cuda::Filter> sobel_filter = cuda::createLaplacianFilter(gpu_frame_1.type(), gpu_frame_1.type(), 3); 

    sobel_filter->apply(gpu_frame_1, gpu_frame_1);
    sobel_filter->apply(gpu_frame_2, gpu_frame_2);

    Ptr<cuda::ORB> feature_extractor = cuda::ORB::create(500);

    vector<KeyPoint> keys_1, keys_2;
    cuda::GpuMat descriptors_1, descriptors_2; 
    feature_extractor->detectAndCompute(gpu_frame_1, noArray(), keys_1, descriptors_1);
    feature_extractor->detectAndCompute(gpu_frame_2, noArray(), keys_2, descriptors_2);

    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);

    vector< vector<DMatch> > matches; 
    matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);

    vector<DMatch> filtered_matches; 
    float ratio = 0.5; 
    for(int i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < matches[i][1].distance*ratio) {
            filtered_matches.push_back(matches[i][0]);
        }
    }

    vector<Point2f> filtered_points_1(filtered_matches.size(), Point2f(0, 0));
    vector<Point2f> filtered_points_2(filtered_matches.size(), Point2f(0, 0)); 
    for(int i = 0; i < filtered_matches.size(); i++) {
        filtered_points_1[i] = keys_1[filtered_matches[i].queryIdx].pt;
        filtered_points_2[i] = keys_2[filtered_matches[i].trainIdx].pt;
    }

    Mat i_mat = (Mat_<double>(3,3) << 73, 0, 400, 0, 73, 400, 0, 0, 1);
    Mat e_mat; 
    e_mat = findEssentialMat(filtered_points_1, filtered_points_2, i_mat, RANSAC);

    cout << e_mat << "\n" <<endl;

    Mat rot_mat, t_vec; 
    recoverPose(e_mat, filtered_points_1, filtered_points_2, rot_mat, t_vec);

    cout << rot_mat << "\n" << endl;
    cout << t_vec << endl; 

    cout << endl;

    vector< vector<float> > rotations  {{rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),},
                                        {rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),},
                                        {rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),}};


    // for(int i = 0; i < rotations.size(); i++) {
    //     for(int j = 0; j < rotations[0].size(); j++)
    //         cout << rotations[i][j] << " ";
    //     cout << endl;
    // }
        
    AngleConverter converter; 
    converter.SetAndConvertR(rotations);
    
    vector<float> angles(3, 0);
    angles = converter.GetYawPitchRollSTD();

    for(int i = 0; i < angles.size(); i++)
        cout << angles[i]*360/3.14 << " ";

    cout << endl; 

    // gpu_frame_1.download(frame_1);
    TileDepthmeter depthmeter; 
    depthmeter.SetFrame(frame_1);
    depthmeter.SetTileSize(0.05, 0.15);
    vector< vector<float> > camera_specs {{73, 0, 400},
                                        {0, 73, 400},
                                        {0, 0, 1}};
    depthmeter.SetCameraIntrinsics(camera_specs);
    depthmeter.SetPreprocessingParams(Size(5, 5), 9, 10, 10, 350);
    depthmeter.SetFilterParams(2, 1000, 10, 5);
    depthmeter.PreprocessImage(); 

    cout << "dst: " << depthmeter.CalcDistance() << endl; 

    cout << endl; 
    
    cout << depthmeter.CalcScale() << endl;

    cout << endl; 
    
    cout << depthmeter.CalcScale()*t_vec << endl;


    // float val = Pi/360; 
    // transform(yaw_pitch_roll.begin(), yaw_pitch_roll.end(), yaw_pitch_roll.begin(), [val](int &element){return element*val;});

    // Mat matches_img; 
    // vector<KeyPoint> keys_1, keys_2;

    // detector->convert(gpu_keys_1, keys_1);
    // detector->convert(gpu_keys_1, keys_2);

    // Mat matches_img; 
    // drawMatches(frame_1, keys_1, frame_2, keys_2, matches, matches_img);
    // Mat filtered_matches_img; 
    // drawMatches(frame_1, keys_1, frame_2, keys_2, filtered_matches, filtered_matches_img);
    // Mat blurred_img;
    // GaussianBlur(frame_1, blurred_img, cv::Size(0, 0), 3);
    // addWeighted(frame_1, 1.5, blurred_img, -1, 0, blurred_img);

    gpu_frame_1.download(frame_1);
    gpu_frame_2.download(frame_2);

    imshow("Frame 1", frame_1);
    imshow("Frame 2", frame_2);
    imshow("TD: ", depthmeter.pipeline.frame_laplacian);
    // imshow("Matches", matches_img);
    // imshow("Filtered Matches", filtered_matches_img);
    waitKey(0);
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