#include <iostream>
#include <ctime>
#include <numeric>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

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

int main(int, char**) {


    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_4.avi");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }


    Mat frame_1, frame_2; 
    cap.set(CAP_PROP_POS_MSEC, 0);
    cap >> frame_1;

    cuda::GpuMat gpu_frame_1;
    cuda::GpuMat gpu_frame_2; 
    cuda::GpuMat gpu_frame_filtered;

    gpu_frame_2.upload(frame_1);
    cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_16U, CV_16U, Size(7, 7), 1, 1);
    Ptr<cuda::Filter> laplace_filter = cuda::createLaplacianFilter(CV_16U, CV_16U, 1); 
    cuda::SURF_CUDA surf(100, 4, 2, true);

    vector<KeyPoint> keys_1, keys_2;
    cuda::GpuMat gpu_keys_1, gpu_keys_2;

    vector<float> descriptors_1, descriptors_2; 
    cuda::GpuMat gpu_descriptors_1, gpu_descriptors_2; 

    gpu_frame_1 = gpu_frame_2; 
    surf(gpu_frame_2, cuda::GpuMat(), gpu_keys_2, gpu_descriptors_2);

    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);

    clock_t start, end; 
    for(int iter = 1; iter < 100+1; iter++) {
        gpu_keys_1 = gpu_keys_2; 
        gpu_descriptors_1 = gpu_descriptors_2;
        gpu_frame_1 = gpu_frame_2; 
        cap >> frame_2;
        

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

        surf(gpu_frame_2, cuda::GpuMat(), gpu_keys_2, gpu_descriptors_2);

        // vector< vector<DMatch> > matches; 
        // matcher->knnMatch(gpu_descriptors_1, gpu_descriptors_2, matches, 2);

        vector< vector<DMatch> > matches; 
        matcher->knnMatch(gpu_descriptors_1, gpu_descriptors_2, matches, 2);

        vector<DMatch> filtered_matches; 
        float ratio = 0.01; 
        for(int i = 0; i < matches.size(); i++) {
            if(matches[i][0].distance < matches[i][1].distance*ratio) {
                filtered_matches.push_back(matches[i][0]);
            }
        }

        surf.downloadKeypoints(gpu_keys_1, keys_1);
        surf.downloadKeypoints(gpu_keys_2, keys_2);

        // vector<Point2f> filtered_points_1(filtered_matches.size(), Point2f(0, 0));
        // vector<Point2f> filtered_points_2(filtered_matches.size(), Point2f(0, 0)); 
        // for(int i = 0; i < filtered_matches.size(); i++) {
        //     filtered_points_1[i] = keys_1[filtered_matches[i].queryIdx].pt;
        //     filtered_points_2[i] = keys_2[filtered_matches[i].trainIdx].pt;
        // }
        Mat i_mat = (Mat_<double>(3,3) << 73, 0, 400, 0, 73, 400, 0, 0, 1);
        Mat h_mat;
        vector<Mat> rot_mat, t_vec, normies; 
        Mat ang, norm_rot_mat;

        try {
            std::vector<Point2f> obj;
            std::vector<Point2f> scene;
            for( size_t i = 0; i < filtered_matches.size(); i++ ) {
                //-- Get the keypoints from the good matches
                obj.push_back( keys_1[ filtered_matches[i].queryIdx ].pt );
                scene.push_back( keys_2[ filtered_matches[i].trainIdx ].pt );
            }
            h_mat = findHomography( obj, scene, RANSAC , 0.01, noArray());
            decomposeHomographyMat(h_mat, i_mat, rot_mat, t_vec, normies); 
            norm_rot_mat = (Mat_<double>(3, 3) << rot_mat[0].at<double>(0,0), rot_mat[0].at<double>(1,0), rot_mat[0].at<double>(2,0),
                                                  rot_mat[0].at<double>(3,0), rot_mat[0].at<double>(4,0), rot_mat[0].at<double>(5,0),
                                                  rot_mat[0].at<double>(6,0), rot_mat[0].at<double>(7,0), rot_mat[0].at<double>(8,0));
            Rodrigues(norm_rot_mat, ang, noArray());
            cout << ang*360/CV_PI << endl; 
        }
        catch(...){
            cout << "Problem!" << endl; 
        }

        // decomposeHomographyMat(h_mat, i_mat, rot_mat, t_vec, normies); 

        // Mat i_mat = (Mat_<double>(3,3) << 73, 0, 400, 0, 73, 400, 0, 0, 1);
        // Mat h_mat, rot_mat, t_vec; node + launch 

        // h_mat = findHomography(filtered_points_1, filtered_points_2, RANSAC);

        // decomposeHomographyMat(h_mat, i_mat, rot_mat, t_vec, noArray()); 

        // cout << rot_mat << endl; 
        // cout << t_vec << endl; 

        // vector< vector<float> > rotations  {{rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),},
        //                                     {rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),},
        //                                     {rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),}};
            
        // AngleConverter converter; 
        // converter.SetAndConvertR(rotations);
    

        // vector<float> angles(3, 0);
        // angles = converter.GetYawPitchRollSTD();

        // for(int i = 0; i < angles.size(); i++)
        //     cout << angles[i]*360/3.14 << " ! ";

        // cout << endl; 

        // try {                    
        //     // e_mat = findEssentialMat(filtered_points_1, filtered_points_1, i_mat, RANSAC, 0.9, 0.1, 10000, noArray());

        //     // cout << e_mat << "\n" << endl; 
 
        //     // recoverPose(e_mat, filtered_points_1, filtered_points_1, i_mat, rot_mat, t_vec);

        //     // cout << rot_mat << "\n" << endl;

        //     // cout << t_vec << endl; 

        //     // cout << endl;
        //     h_mat = cv::findHomography(filtered_points_1, filtered_points_2, RANSAC);

        //     decomposeHomographyMat(h_mat, i_mat, rot_mat, t_vec, noArray()); 

        //     cout << rot_mat << endl; 
        //     cout << t_vec << endl; 

        //     vector< vector<float> > rotations  {{rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),},
        //                                         {rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),},
        //                                         {rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),}};
                
        //     AngleConverter converter; 
        //     converter.SetAndConvertR(rotations);
        

        //     vector<float> angles(3, 0);
        //     angles = converter.GetYawPitchRollSTD();

        //     for(int i = 0; i < angles.size(); i++)
        //         cout << angles[i]*360/3.14 << " ! ";

        //     cout << endl; 
        // }
        // catch(...) {
        //     cout << "Oops!" << endl;
        // }


        // float val = Pi/360; 
        // transform(yaw_pitch_roll.begin(), yaw_pitch_roll.end(), yaw_pitch_roll.begin(), [val](int &element){return element*val;});

        // Mat matches_img; 
        // vector<KeyPoint> keys_1, keys_2;

        // detector->convert(gpu_keys_1, keys_1);
        // detector->convert(gpu_keys_1, keys_2);

        // Mat matches_img; 
        // drawMatches(frame_1, keys_1, frame_2, keys_2, matches, matches_img);

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);

        Mat filtered_matches_img; 
        drawMatches(frame_1, keys_1, frame_2, keys_2, filtered_matches, filtered_matches_img);
        // Mat blurred_img;
        // GaussianBlur(frame_1, blurred_img, cv::Size(0, 0), 3);
        // addWeighted(frame_1, 1.5, blurred_img, -1, 0, blurred_img);

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
// pochette 19100 no 
//petite boite 79k 2-14c weeks 
//mini boite 35k emaar
//souple 43500 brown 46k black 