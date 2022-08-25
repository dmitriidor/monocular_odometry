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


    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_3_cut.mp4");

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
    // cuda::SURF_CUDA surf(1500, 4, 2, true);
    Ptr<cuda::ORB> feature_extractor = cuda::ORB::create(100, 1.2f, 8, 40, 0, 2, 0, 50);

    vector<KeyPoint> keys_1, keys_2;
    cuda::GpuMat gpu_keys_1, gpu_keys_2;

    // vector<float> descriptors_1, descriptors_2; 
    cuda::GpuMat gpu_descriptors_1, gpu_descriptors_2; 
    gpu_frame_2.convertTo(gpu_frame_2, CV_32FC1);
    cuda::normalize(gpu_frame_2, gpu_frame_2, 0, 1, NORM_MINMAX, CV_32FC1);

    gpu_frame_1 = gpu_frame_2; 


    float angle = 0; 
    clock_t start, end; 
    for(int iter = 1; iter < 1000000+1; iter++) {
        // cap.set(CAP_PROP_POS_MSEC, 1*iter);
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
        gpu_frame_2.convertTo(gpu_frame_2, CV_32FC1);
        cuda::normalize(gpu_frame_2, gpu_frame_2, 0, 1, NORM_MINMAX, CV_32FC1);

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);
        Mat framess_1 = frame_1;
        Mat framess_2 = frame_2; 

        // double minVal, maxVal;
        // Point2i minLoc, maxLoc;
        // minMaxLoc(frame_1, &minVal, &maxVal, &minLoc, &maxLoc);
        // frame_1.convertTo(frame_1, CV_32FC1, 65535.0/(maxVal-minVal),-65535.0*minVal/(maxVal-minVal)); 
        // minMaxLoc(frame_2, &minVal, &maxVal, &minLoc, &maxLoc);
        // frame_2.convertTo(frame_2, CV_32FC1, 65535.0/(maxVal-minVal),-65535.0*minVal/(maxVal-minVal)); 

        float min_x = frame_1.cols/2; 
        float min_y = frame_1.rows/2;
        linearPolar(frame_1, frame_1, Point2f(min_x, min_y), min((int)min_x, (int)min_y), INTER_NEAREST);
        linearPolar(frame_2, frame_2, Point2f(min_x, min_y), min((int)min_x, (int)min_y), INTER_NEAREST);

        Point2d shift = phaseCorrelate(frame_2, frame_1); 

        angle += -shift.y/min_x * 360; 
        cout << "shift: " << -shift.y/min_y * 360 << endl;
        cout << "angle: " << angle << "\n" << endl; 

        imshow("1st", framess_1);
        imshow("2nd", framess_2);

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
