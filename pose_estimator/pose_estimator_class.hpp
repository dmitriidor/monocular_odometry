#ifndef POSE_ESTIMATOR_CLASS_H
#define POSE_ESTIMATOR_CLASS_H

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

using namespace cv;
using namespace std; 
using namespace Eigen; 
class PoseEstimator{
    public:
    PoseEstimator() = default; 
    void SetPrevFrame(Mat frame); 
    void SetNextFrame(Mat frame); 
    void SetGaussParams(int stype, int dtype, Size k, float s_x, float s_y);
    void SetLaplaceParams(int stype, int dtype, int k); 
    void SetORBParams(int nftrs, float scl, int nlvls, int edgeT, int firstLvl, int WTA_K, int sType, int pSize, int fastT, bool blurD);
    void SetRatioTestParams(float ratio);
    void SetRegionTestParams(float distance); 
    Mat GetProcessedFrame(); 
    vector<double> GetYPR();
    vector<double> GetQ();
    vector<double> GetT();
    bool CalculateMotion(); 

    private:
    void SharpenImage();
    void ExtractFeatures(); 
    void ExtractMatches();
    void FilterMatchesRatio(); 
    void FilterMatchesRegion();

    void Convert3D(); 
    void CalcNextCentroid(); 
    void CenterSet(); 
    void ExtractRotation();
    void CheckRCase(); 
    void Convert2YPR(); 
    void Convert2Q(); 
    void CalculateTranslation(); 
    void CycleData(); 

    Mat frame_prev, frame_next; 
    cuda::GpuMat gpu_frame_prev, gpu_frame_next, gpu_frame_mask;
    
    struct Gaussian{
        int src_type = CV_8U; 
        int dst_type = CV_8U;
        Size ksize = Size(7, 7);
        float sigma_x = 1;
        float sigma_y = 1; 
    };
    Gaussian pGau; 

    struct Laplace{
        int src_type = CV_8U; 
        int dst_type = CV_8U;
        int ksize = 1; 
    };
    Laplace pLap; 

    struct ORB{
        int nfeatures = 3000; 
        float scale = 1.2;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int scoreType = 0;
        int patchSize = 31;
        int fastThreshold = 20;
        bool blurForDescriptor = false;
    }; 
    ORB pORB;

    float filter_ratio_ = 0.8; 
    float filter_distance_ = 20; 

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(pGau.src_type, pGau.dst_type, pGau.ksize, pGau.sigma_x, pGau.sigma_y);
    Ptr<cuda::Filter> laplacian_filter = cuda::createLaplacianFilter(pLap.src_type, pLap.dst_type, pLap.ksize); 

    Ptr<cuda::ORB> feature_extractor = cuda::ORB::create(pORB.nfeatures, pORB.scale, 
            pORB.nlevels, pORB.edgeThreshold, pORB.firstLevel, pORB.WTA_K, pORB.scoreType, pORB.patchSize);

    vector<KeyPoint> keys_prev, keys_next;
    cuda::GpuMat gpu_descriptors_prev, gpu_descriptors_next; 

    vector< vector<DMatch> > matches;
    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);

    vector<DMatch> matches_ratioed, matches_regioned, matches_filtered; 

    MatrixXd points3d_prev, points3d_next; 
    Vector3d centroid_prev, centroid_next;
    Matrix3d H, R;
    Vector3d YPR, T; 
    Quaterniond Q; 
};


#endif