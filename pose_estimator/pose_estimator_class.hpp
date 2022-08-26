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
    Mat GetProcessedFrame(); 
    vector<double> GetYPR();
    vector<double> GetQ();
    vector<double> GetT();
    bool CalculateMotion(); 

    private:
    void SharpenImage();

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

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(pGau.src_type, pGau.dst_type, pGau.ksize, pGau.sigma_x, pGau.sigma_y);
    Ptr<cuda::Filter> laplacian_filter = cuda::createLaplacianFilter(pLap.src_type, pLap.dst_type, pLap.ksize); 

    MatrixXd points3d_prev, points3d_next; 
    Vector3d centroid_prev, centroid_next;
    Matrix3d H, R;
    Vector3d YPR, T; 
    Quaterniond Q; 
};


#endif