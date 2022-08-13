#ifndef POSE_ESTIMATOR_CLASS_H
#define POSE_ESTIMATOR_CLASS_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include <opencv4/opencv2/core/cuda.hpp>


using namespace cv;

class PoseEstimator{
    public:
    PoseEstimator(); 
    // void SetFilterParams(int sample_size, int cycles, float thrshld, int min_inliers);
    // void SetPreprocessingParams(Size kernel, int laplace_krnl, int bin_thrshld, float thrshld_1, float thrshld_2);
    // void SetFrame(Mat new_frame); 

    // void PreprocessImage();
    // float CalcVelocities(); 

    // struct FramePipeline { 
    //     Mat frame_raw; 
    //     Mat frame_blurred_1;
    //     Mat frame_gray; 
    //     Mat frame_laplacian; 
    //     Mat frame_blurred_2; 
    //     Mat frame_bin; 
    //     Mat frame_edges; 
    // };
    // FramePipeline pipeline;
    
    // private:
    // void DetectFAST(); 
    // void ExtractBEBLID(); 
    // void FilterRatio();
    // void CalcRelPose();
    // void CalcSpeeds();  

    // Mat crnt_frame, prev_frame; 

    // Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
    // vector<KeyPoint> crnt_keys, prev_keys;

    // Ptr<Feature2D> extractor = xfeatures2d::BEBLID::create(0.75);
    // Mat crnt_descriptors, prev_descriptors; 

    // BFMatcher matcher;
    // vector<vector<DMatch>> matches;

    // vector<DMatch> matches_1, matches_2;
    // vector<DMatch> filtered_matches;

    // vector<Point2f> crnt_matched, prev_matched;

    // Mat E, R, T;

    // struct PreprocessingParams {
    //     Size gaussian_kernel;
    //     int bin_threshold;
    //     int laplacian_kernel;
    //     int canny_threshold_1;
    //     int canny_threshold_2;
    // };
    // PreprocessingParams pp_vals; 

    // float ratio_filter_thresh_ = 0.8; 
    // float scale_factor_ = 0; 
};


#endif