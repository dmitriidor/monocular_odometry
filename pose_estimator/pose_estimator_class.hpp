#ifndef POSE_ESTIMATOR_CLASS_H
#define POSE_ESTIMATOR_CLASS_H

#include <numeric>
#include <eigen3/Eigen/Dense>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

class PoseEstimator{
    public:
    PoseEstimator() = default; 

    void SetPrevFrame(cv::Mat frame); 

    void SetNextFrame(cv::Mat frame); 

    void SetGaussParams(int stype, 
                        int dtype, 
                        cv::Size k, 
                        float s_x, 
                        float s_y);

    void SetLaplaceParams(int stype, 
                          int dtype, 
                          int k); 

    void SetORBParams(int nftrs, 
                      float scl, 
                      int nlvls, 
                      int edgeT, 
                      int firstLvl, 
                      int WTA_K,
                      int sType, 
                      int pSize, 
                      int fastT, 
                      bool blurD);

    void SetRANSACParams(int iterations, 
                         int sample_size,
                         int min_score);

    void SetRatioTestParams(float ratio);

    void SetRegionTestParams(float distance); 

    void SetRansacParams(int num_of_iters,
                         int sample_size, 
                         int min_score,
                         float threshold); 

    cv::Mat GetProcessedFrame(); 

    std::vector<double> GetYPR();

    std::vector<double> GetQ();

    std::vector<double> GetT();

    std::string GetErrorMsg(); 

    void ShowMatches(); 

    bool CalculateMotion(); 

    private:
    void SharpenImage(cv::cuda::GpuMat& gpu_frame);

    void ExtractFeatures(const cv::cuda::GpuMat& frame, 
                         cv::cuda::GpuMat& gpu_descriptors, 
                         std::vector<cv::KeyPoint>& keys); 

    void ExtractMatchesA2B(const cv::cuda::GpuMat& gpu_descriptors_A, 
                           const cv::cuda::GpuMat& gpu_descriptors_B, 
                           std::vector< std::vector<cv::DMatch> >& matchesA2B);

    void FilterMatchesRatio(const std::vector< std::vector<cv::DMatch> >& knn_matches, 
                            std::vector<cv::DMatch>& filtered_matches); 

    void FilterSymmetry(const std::vector<cv::DMatch>& matchesA2B, 
                        const std::vector<cv::DMatch>& matchesB2A,
                        std::vector<cv::DMatch>& filtered_matches); 

    void FilterMatchesRegion(const std::vector<cv::DMatch>& matches, 
                             std::vector<cv::DMatch>& filtered_matches, 
                             std::vector<cv::KeyPoint>& keys_A, 
                             std::vector<cv::KeyPoint>& keys_B);

    void ConvertPoints3D(const std::vector<cv::DMatch>& matches,
                         const std::vector<cv::KeyPoint>& keys,
                         Eigen::MatrixXd& points_3d); 

    double CalcRT_RANSAC(const Eigen::MatrixXd& points_3d_A, 
                         const Eigen::MatrixXd& points_3d_B, 
                         Eigen::Matrix3d& rot_mat,
                         Eigen::Vector3d& t_vec); 

    void FitLSR(Eigen::MatrixXd& points_3d_A, 
                Eigen::MatrixXd& points_3d_B,
                Eigen::Matrix3d& rot_mat,
                Eigen::Vector3d& t_vec);

    void CalcCentroid(const Eigen::MatrixXd& points_3d, Eigen::Vector3d& centroid); 

    void CenterSet(const Eigen::Vector3d& centroid, Eigen::MatrixXd& points_3d); 

    void ExtractRotationA2B(const Eigen::MatrixXd& points_3d_centered_A,
                            const Eigen::MatrixXd& points_3d_centered_B,
                            Eigen::Matrix3d& rot_mat);

    void ExtractTranslationA2B(const Eigen::Matrix3d& rot_mat,
                               const Eigen::Vector3d& centroid_A,
                               const Eigen::Vector3d& centroid_B, 
                               Eigen::Vector3d& t_vec);

    void CheckRCase(Eigen::Matrix3d& rot_mat); 

    void ConvertR2YPR(const Eigen::Matrix3d& rot_mat, Eigen::Vector3d& angles); 

    void ConvertR2Q(const Eigen::Matrix3d& rot_mat, Eigen::Quaterniond& quat);  

    void CycleData(); 

    void HandleError(); 
    
    struct Gaussian{
        int src_type = CV_8U; 
        int dst_type = CV_8U;
        cv::Size ksize = cv::Size(7, 7);
        float sigma_x = 1;
        float sigma_y = 1; 
    };
    Gaussian pGau_; 

    struct Laplace{
        int src_type = CV_8U; 
        int dst_type = CV_8U;
        int ksize = 1; 
    };
    Laplace pLap_;

    struct ORB{
        int nfeatures = 1000; 
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
    ORB pORB_;

    struct RANSAC{
        int iterations = 1000; 
        int sample_size = 3;
        int min_score = 20; 
        float threshold = 200; 
    }; 
    RANSAC pR_; 

    struct ProcessingStep{
        cv::Mat frame;
        cv::cuda::GpuMat gpu_frame, gpu_descriptors;
        std::vector<cv::KeyPoint> keys;
        Eigen::MatrixXd points_3d; 
    };
    ProcessingStep prev_, next_; 

    float filter_ratio_ = 0.8; 
    float filter_distance_ = 20; 

    cv::Ptr<cv::cuda::Filter> gaus_filter_ = cv::cuda::createGaussianFilter(pGau_.src_type, 
                                                                            pGau_.dst_type, 
                                                                            pGau_.ksize, 
                                                                            pGau_.sigma_x, 
                                                                            pGau_.sigma_y);
                                                                            
    cv::Ptr<cv::cuda::Filter> lplc_filter_ = cv::cuda::createLaplacianFilter(pLap_.src_type, 
                                                                             pLap_.dst_type, 
                                                                             pLap_.ksize); 

    cv::Ptr<cv::cuda::ORB> feature_extractor_ = cv::cuda::ORB::create(pORB_.nfeatures, 
                                                                      pORB_.scale, 
                                                                      pORB_.nlevels, 
                                                                      pORB_.edgeThreshold, 
                                                                      pORB_.firstLevel, 
                                                                      pORB_.WTA_K,
                                                                      pORB_.scoreType, 
                                                                      pORB_.patchSize);

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    std::vector<cv::DMatch> filtered_matches_, matches_prev2next_, matches_next2prev_;
    std::vector< std::vector<cv::DMatch> > knn_matches_;

    Eigen::Matrix3d R_;
    Eigen::Vector3d YPR_, T_; 
    Eigen::Quaterniond Q_; 

    enum ErrorTypes{
        NO_ERRORS,
        NO_MATCHES_FOUND, 
        NOT_ENOUGH_MATCHES,
        NO_GOOD_SOLUTION_RANSAC
    };
    ErrorTypes error_; 
    std::string error_message_; 
};


#endif