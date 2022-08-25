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
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/xfeatures2d/cuda.hpp>
#include <opencv4/opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/cudalegacy.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "pose_estimator/pose_estimator_class.hpp"
#include "angle_converter/angle_converter_class.hpp"
#include "tile_depthmeter/tile_depthmeter_class.hpp"

using namespace std;
using namespace cv;

int main(int, char**) {

    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_3_cut.mp4");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame_1, frame_2; 
    cap.set(CAP_PROP_POS_MSEC, 1000);
    cap >> frame_1;

    cuda::GpuMat gpu_frame_1;
    cuda::GpuMat gpu_frame_2; 
    cuda::GpuMat gpu_frame_flow;

    gpu_frame_2.upload(frame_1);
    cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_16U, CV_16U, Size(5, 5), 1, 1);
    Ptr<cuda::Filter> laplace_filter = cuda::createLaplacianFilter(CV_16U, CV_16U, 1); 

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    int loops = 100; 
    Ptr<cuda::FarnebackOpticalFlow> flow = cuda::FarnebackOpticalFlow::create(5, 0.5, false);

    Rect region(50, 50, 300, 300);
    cuda::GpuMat gpu_matching_region_1, gpu_matching_region_2;
    Mat frame_flow, matched_frame_1, matched_frame_2; 

    gpu_frame_2.convertTo(gpu_frame_2, CV_16U, 1/pow(2, 8));
    gpu_matching_region_2 = gpu_frame_2(region);

    cuda::GpuMat gpu_status, gpu_error; 

    for(int iter = 1; iter < 10000+1; iter++) {

        gpu_matching_region_1 = gpu_matching_region_2; 
        gpu_frame_1 = gpu_frame_2; 
        cap >> frame_2;

        gpu_frame_2.upload(frame_2); 
        // gpu_frame_filtered.upload(frame_2);
        cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);
        // cuda::cvtColor(gpu_frame_filtered, gpu_frame_filtered, COLOR_BGR2GRAY);
        // gpu_frame_filtered.convertTo(gpu_frame_filtered, CV_16U);
        // cuda::normalize(gpu_frame_filtered, gpu_frame_filtered, 0, pow(2, 16), NORM_MINMAX, CV_16U);
        // gaussian_filter->apply(gpu_frame_filtered, gpu_frame_filtered);
        // laplace_filter->apply(gpu_frame_filtered, gpu_frame_filtered); 
        // double minimum_value, maximum_value; 
        // cuda::minMax(gpu_frame_filtered, &minimum_value, &maximum_value);  
        gpu_frame_2.convertTo(gpu_frame_2, CV_16U);
        cuda::normalize(gpu_frame_2, gpu_frame_2, 0, pow(2, 16), NORM_MINMAX, CV_16U);
        // cuda::subtract(gpu_frame_2, gpu_frame_filtered, gpu_frame_2);
        // cuda::minMax(gpu_frame_2, &minimum_value, &maximum_value); 
        gpu_matching_region_2 = gpu_frame_2(region);
        

        flow->calc(gpu_matching_region_1, gpu_matching_region_2, gpu_frame_flow); 

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);

        gpu_matching_region_1.download(matched_frame_1);
        gpu_matching_region_2.download(matched_frame_2);

        gpu_frame_flow.download(frame_flow); 

        Mat flow_parts[2];
        split(frame_flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);

        imshow("Dense", bgr);
        imshow("Frame 1", matched_frame_1);
        imshow("Frame 2", matched_frame_2);
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
