#ifndef POSE_ESTIMATOR_CLASS_H
#define POSE_ESTIMATOR_CLASS_H

#include <eigen3/Eigen/Dense>

#include <opencv4/opencv2/videoio.hpp>

#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

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
    void SetMatcherParams(int number_of_cells, int border_size, int max_spread, int stype, int m, Size ks); 
    Mat GetProcessedFrame(); 
    vector<double> GetYPR();
    vector<double> GetQ();
    vector<double> GetT();
    bool CalculateMotion(); 

    private:
    void SharpenImage();
    void SquareImage();
    void SetMatrix();
    void ExtractMotion(); 
    void FilterDeltas(); 
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
    cuda::GpuMat gpu_subregion_to_match, gpu_region_for_matching; 
    cuda::GpuMat gpu_matching_results; 
    Mat matching_results; 
    
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

    struct TMatcher{ 
        int number_of_cells = 16; 
        int border_size = 10; 
        int max_spread = 1; 

        int src_type = CV_8U; 
        int method = TM_CCOEFF; 
        Size size = Size(0, 0); 
    };
    TMatcher pMat; 

    Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(pGau.src_type, pGau.dst_type, pGau.ksize, pGau.sigma_x, pGau.sigma_y);
    Ptr<cuda::Filter> laplacian_filter = cuda::createLaplacianFilter(pLap.src_type, pLap.dst_type, pLap.ksize); 
    Ptr<cuda::TemplateMatching> matcher = cuda::createTemplateMatching(pMat.src_type, pMat.method, pMat.size);

    MatrixXi chosen_cells; 
    MatrixXd deltas, positions; 
    MatrixXd deltas_filtered, positions_filtered; 
    MatrixXd points3d_prev, points3d_next; 
    Vector3d centroid_prev, centroid_next;
    Matrix3d H, R;
    Vector3d YPR, T; 
    Quaterniond Q; 
};


#endif