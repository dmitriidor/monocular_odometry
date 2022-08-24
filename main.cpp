#include <iostream>
#include <ctime>
#include <numeric>

#include <eigen3/Eigen/Dense>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

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


    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_2.avi");

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

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);

        Mat dst, cdst, cdstP;
        Canny(frame_1, frame_1, 50, 200, 3);
        // threshold(frame_1, frame_1, 100, 255, THRESH_BINARY_INV);
        // Sobel(frame_1, frame_1, CV_8U, 1, 1, 5);
        dilate(frame_1, frame_1, Mat(), Point(-1, -1), 1, 1, 1);
        // erode(frame_1, frame_1, Mat(), Point(-1, -1), 3, 1, 1);
        // cvtColor(frame_1, cdst, COLOR_GRAY2BGR);
        cdst = Mat::zeros(frame_1.rows, frame_1.cols, CV_8UC1);
        cdstP = Mat::zeros(frame_1.rows, frame_1.cols, CV_8UC1);

        vector<Vec2f> lines; // will hold the results of the detection
        HoughLines(frame_1, lines, 1, CV_PI/180, 200, 50, 0 ); // runs the actual detection
        // Draw the lines

        float rho_threshold = 60;
        float theta_threshold = CV_PI/360*30;
        vector<vector<int>> similar_lines;
        // CV_PI/360*20
        for(int i = 0; i < lines.size(); i++){
            vector<int> temp {i, 1}; 
            similar_lines.push_back(temp);
            for(int j = i; j < lines.size(); j++){
                if(i == j)
                    continue; 
                
                if(abs(lines[i][0] - lines[j][0]) < rho_threshold && abs(lines[i][1] - lines[j][1]) < theta_threshold){
                    similar_lines.back()[1]++;
                }
            }
        }
        
        vector<bool> flags (similar_lines.size(), true); 
        sort(similar_lines.begin(), similar_lines.end(), [](auto const& lhs, auto const& rhs) {
            return lhs[1] > rhs[1];
            });

        vector<Vec2f> filtered_lines;
        for(int i = 0; i < similar_lines.size(); i++){
            if(!flags[i])
                continue;
            
            float rho_avg = 0; 
            float theta_avg = 0;
            int similars = 0; 
            for(int j = i+1; j < similar_lines.size(); j++){
                if(!flags[j])
                    continue;
                
                if(abs(lines[i][0] - lines[j][0]) < rho_threshold && abs(lines[i][1] - lines[j][1]) < theta_threshold){
                    flags[j] = false; 
                    rho_avg += lines[j][0];
                    theta_avg += lines[j][1];
                    similars++; 
                }
            }
            filtered_lines.push_back(Vec2f(rho_avg/similars, theta_avg/similars));
        }

        
        
        cout << "Lines: " << lines.size() << endl;
        cout << "After Filtering: " << filtered_lines.size() << endl;

        for( size_t i = 0; i < filtered_lines.size(); i++ ){
            float rho = filtered_lines[i][0], theta = filtered_lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( cdst, pt1, pt2, 255, 3, LINE_AA);
        }
   
        imshow("Original", frame_1);
        imshow("Res", cdst);
        // imshow("TD: ", depthmeter.pipeline.frame_laplacian);
        // imshow("Matches", matches_img);
        // imshow("Filtered Matches", filtered_matches_img);

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
