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

static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}
static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8U, (void*)&vec[0]);
    d_mat.download(mat);
}
static void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;
            Point p = prevPts[i];
            Point q = nextPts[i];
            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);
            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );
            if (hypotenuse < 1.0)
                continue;
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));
            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);
            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.
            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

int main(int, char**) {


    VideoCapture cap("/home/dmitrii/Downloads/vids/archive/pool_test_2.avi");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }


    Mat frame_1, frame_2; 
    cap.set(CAP_PROP_POS_MSEC, 1000);
    cap >> frame_1;

    cuda::GpuMat gpu_frame_1;
    cuda::GpuMat gpu_frame_2; 
    cuda::GpuMat gpu_frame_filtered;

    gpu_frame_2.upload(frame_1);
    cuda::cvtColor(gpu_frame_2, gpu_frame_2, COLOR_BGR2GRAY);

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_16U, CV_16U, Size(5, 5), 1, 1);
    Ptr<cuda::Filter> laplace_filter = cuda::createLaplacianFilter(CV_16U, CV_16U, 1); 

    Mat corners_1, corners_2; 
    cuda::GpuMat gpu_corners_1, gpu_corners_2; 
    Ptr<cuda::CornersDetector> detector= cuda::createGoodFeaturesToTrackDetector(CV_16U, 100, 0.1, 10, 5, false);

    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    int loops = 100; 
    Ptr<cuda::SparsePyrLKOpticalFlow> flow = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, loops);

    Rect region(50, 50, 300, 300);
    cuda::GpuMat gpu_matching_region_1, gpu_matching_region_2;

    Mat matched_frame_1, matched_frame_2; 

    gpu_frame_2.convertTo(gpu_frame_2, CV_16U, 1/pow(2, 8));
    gpu_matching_region_2 = gpu_frame_2(region);
    detector->detect(gpu_matching_region_2, gpu_corners_2);

    cuda::GpuMat gpu_status, gpu_error; 

    for(int iter = 1; iter < 100+1; iter++) {

        gpu_matching_region_1 = gpu_matching_region_2; 
        gpu_corners_1 = gpu_corners_2; 
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
        
        if(iter%10 == 0)
            detector->detect(gpu_matching_region_2, gpu_corners_1);

        flow->calc(gpu_matching_region_1, gpu_matching_region_2, gpu_corners_1, gpu_corners_2, gpu_status, gpu_error); 

        gpu_frame_1.download(frame_1);
        gpu_frame_2.download(frame_2);

        gpu_corners_1.download(corners_1);
        gpu_corners_2.download(corners_2);

        gpu_matching_region_1.download(matched_frame_1);
        gpu_matching_region_2.download(matched_frame_2);

        vector<uchar> status(gpu_status.cols);
        download(gpu_status, status);
        Mat good_new;

        for(int i = 0; i < corners_1.rows; i++)
        {
            if(status[i] == 1) {
                good_new.push_back(corners_2.at<float>(i, 0));
                good_new.push_back(corners_2.at<float>(i, 1));
                // cout << corners_2.at<float>(i, 1) << endl;
            }
        }

        cout << good_new << endl;
        // corners_2 = good_new; 

        // cout << corners_1 << endl; 

        for(int i = 0; i < corners_1.cols; i++) {
            Point2f center_1 = corners_1.at<Point2f>(0, i);
            circle(matched_frame_1, center_1, 5, 0, 2, 8);
        }

        for(int i = 0; i < corners_2.cols; i++) {
            Point2f center_2 = corners_2.at<Point2f>(0, i);
            circle(matched_frame_2, center_2, 5, 100, 2, 8);
        }

        drawArrows(frame_1, corners_1, corners_2, status, 255);
        imshow("PyrLK [Sparse]", frame_1);


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
