#include "pose_estimator_class.hpp"

void PoseEstimator::SetPrevFrame(cv::Mat frame){
    prev_.frame = frame; 
    prev_.gpu_frame.upload(frame);

    SharpenImage(prev_.gpu_frame);
    ExtractFeatures(prev_.gpu_frame, prev_.gpu_descriptors, prev_.keys); 
}

void PoseEstimator::SetNextFrame(cv::Mat frame){
    next_.frame = frame; 
    next_.gpu_frame.upload(frame);

    SharpenImage(next_.gpu_frame);
    ExtractFeatures(next_.gpu_frame, next_.gpu_descriptors, next_.keys); 
}

void PoseEstimator::SetGaussParams(int stype, int dtype, cv::Size k, float s_x, float s_y){
    pGau_.src_type = stype; 
    pGau_.dst_type = dtype; 
    pGau_.ksize = k;
    pGau_.sigma_x = s_x;
    pGau_.sigma_y = s_y; 

    cv::Ptr<cv::cuda::Filter> gaus_filter_ = cv::cuda::createGaussianFilter(pGau_.src_type, 
                                                                            pGau_.dst_type, 
                                                                            pGau_.ksize, 
                                                                            pGau_.sigma_x, 
                                                                            pGau_.sigma_y);
}

void PoseEstimator::SetLaplaceParams(int stype, int dtype, int k){
    pLap_.src_type = stype; 
    pLap_.dst_type = dtype; 
    pLap_.ksize = k; 

    cv::Ptr<cv::cuda::Filter> lplc_filter_ = cv::cuda::createLaplacianFilter(pLap_.src_type, 
                                                                             pLap_.dst_type, 
                                                                             pLap_.ksize);     
}

void PoseEstimator::SetORBParams(int nftrs, float scl, int nlvls, int edgeT, int firstLvl, int WTA_K, int sType, int pSize, int fastT, bool blurD){
    pORB_.nfeatures = nftrs; 
    pORB_.scale = scl;
    pORB_.nlevels = nlvls;
    pORB_.edgeThreshold = edgeT;
    pORB_.firstLevel = firstLvl;
    pORB_.WTA_K = WTA_K;
    pORB_.scoreType = sType;
    pORB_.patchSize = pSize;
    pORB_.fastThreshold = fastT;
    pORB_.blurForDescriptor = blurD;

    cv::Ptr<cv::cuda::ORB> feature_extractor_ = cv::cuda::ORB::create(pORB_.nfeatures, 
                                                                      pORB_.scale, 
                                                                      pORB_.nlevels, 
                                                                      pORB_.edgeThreshold, 
                                                                      pORB_.firstLevel, 
                                                                      pORB_.WTA_K,
                                                                      pORB_.scoreType, 
                                                                      pORB_.patchSize);
}

void PoseEstimator::SetRansacParams(int num_of_iters,
                                    int sample_size, 
                                    int min_score,
                                    float threshold){
    pR_.iterations = num_of_iters; 
    pR_.sample_size = sample_size;
    pR_.min_score = min_score; 
    pR_.threshold = threshold; 
}

void PoseEstimator::SetRatioTestParams(float ratio){
    filter_ratio_ = ratio; 
}

void PoseEstimator::SetRegionTestParams(float distance){
    filter_distance_ = distance; 
}

cv::Mat PoseEstimator::GetProcessedFrame(){
    cv::Mat frame; 
    next_.gpu_frame.download(frame);
    return frame; 
}

std::vector<double> PoseEstimator::GetYPR(){
    std::vector<double> temp{YPR_[0],
                             YPR_[1],
                             YPR_[2]};
    return temp; 
}

std::vector<double> PoseEstimator::GetQ(){
    std::vector<double> temp{Q_.w(),
                             Q_.x(),
                             Q_.y(),
                             Q_.z()};
    return temp; 
}

std::vector<double> PoseEstimator::GetT(){
    std::vector<double> temp{T_[0],
                             T_[1],
                             T_[2]};
    return temp; 
}

std::string PoseEstimator::GetErrorMsg(){
    return error_message_; 
}

void PoseEstimator::ShowMatches(){
    cv::Mat matches_image; 
    cv::drawMatches(prev_.frame, prev_.keys, next_.frame, next_.keys,
     filtered_matches_, matches_image);
    cv::imshow("Class Matches", matches_image);
    cv::waitKey(0);
}

bool PoseEstimator::CalculateMotion(){
    ExtractMatchesA2B(next_.gpu_descriptors, prev_.gpu_descriptors, knn_matches_);
    if(knn_matches_.empty()){error_ = NO_MATCHES_FOUND; goto failure;}
    FilterMatchesRatio(knn_matches_, matches_next2prev_);

    ExtractMatchesA2B(prev_.gpu_descriptors, next_.gpu_descriptors, knn_matches_); 
    if(knn_matches_.empty()){error_ = NO_MATCHES_FOUND; goto failure;}
    FilterMatchesRatio(knn_matches_, matches_prev2next_);

    FilterSymmetry(matches_prev2next_, matches_next2prev_, filtered_matches_); 

    FilterMatchesRegion(filtered_matches_, filtered_matches_, prev_.keys, next_.keys);
    if(filtered_matches_.empty() || filtered_matches_.size()<3){error_ = NOT_ENOUGH_MATCHES; goto failure;}
    ShowMatches(); 

    ConvertPoints3D(filtered_matches_, next_.keys, next_.points_3d);
    ConvertPoints3D(filtered_matches_, prev_.keys, prev_.points_3d); 

    double inlier_ratio; 
    inlier_ratio = CalcRT_RANSAC(next_.points_3d, prev_.points_3d, R_, T_);
    ConvertR2YPR(R_, YPR_);
    ConvertR2Q(R_, Q_); 
    if(inlier_ratio < 50){error_ = NO_GOOD_SOLUTION_RANSAC; goto failure;}

    CycleData();
    error_ = NO_ERRORS; 
    return true;

    failure:
    HandleError(); 
    CycleData();
    return false;
}

void PoseEstimator::SharpenImage(cv::cuda::GpuMat& gpu_frame){
    cv::cuda::GpuMat gpu_fmask; 
    gpu_fmask = gpu_frame.clone(); 

    cv::cuda::cvtColor(gpu_frame, gpu_frame, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(gpu_fmask, gpu_fmask, cv::COLOR_BGR2GRAY);

    gpu_frame.convertTo(gpu_frame, CV_8U);
    gpu_fmask.convertTo(gpu_fmask, CV_8U);

    cv::cuda::normalize(gpu_frame, gpu_frame, 0, pow(2, 8), cv::NORM_MINMAX, CV_8U);
    cv::cuda::normalize(gpu_fmask, gpu_fmask, 0, pow(2, 8), cv::NORM_MINMAX, CV_8U);

    gaus_filter_->apply(gpu_frame, gpu_frame);
    lplc_filter_->apply(gpu_fmask, gpu_fmask); 

    cv::cuda::subtract(gpu_frame, gpu_fmask, gpu_frame);
}

void PoseEstimator::ExtractFeatures(const cv::cuda::GpuMat& gpu_frame, 
                                    cv::cuda::GpuMat& gpu_descriptors, 
                                    std::vector<cv::KeyPoint>& keys){
    feature_extractor_->detectAndCompute(gpu_frame, cv::noArray(), keys, gpu_descriptors);
}

void PoseEstimator::ExtractMatchesA2B(const cv::cuda::GpuMat& gpu_descriptors_A, 
                                      const cv::cuda::GpuMat& gpu_descriptors_B, 
                                      std::vector< std::vector<cv::DMatch> >& matchesA2B){
    matcher_->knnMatch(gpu_descriptors_A, gpu_descriptors_B, matchesA2B, 2);
}

void PoseEstimator::FilterMatchesRatio(const std::vector< std::vector<cv::DMatch> >& knn_matches, 
                                       std::vector<cv::DMatch>& filtered_matches){
    std::vector<cv::DMatch> good_matches;  
    for(int i = 0; i < knn_matches.size(); i++){
        if(knn_matches[i][0].distance < knn_matches[i][1].distance*filter_ratio_){
            good_matches.push_back(knn_matches[i][0]); 
        }
    }
    filtered_matches = good_matches; 
}

void PoseEstimator::FilterSymmetry(const std::vector<cv::DMatch>& matchesA2B, 
                                   const std::vector<cv::DMatch>& matchesB2A,
                                   std::vector<cv::DMatch>& filtered_matches){
    std::vector<cv::DMatch> good_matches; 
    for(int i = 0; i < matchesA2B.size(); i ++){
        for(int j = 0; j < matchesB2A.size(); j++){
            if((matchesA2B[i].queryIdx == matchesB2A[j].trainIdx) 
                &&
                (matchesB2A[j].queryIdx == matchesA2B[i].trainIdx)){
                good_matches.push_back(matchesA2B[i]);
                break; 
            }
        }
    }
    filtered_matches = good_matches; 
}

void PoseEstimator::FilterMatchesRegion(const std::vector<cv::DMatch>& matches, 
                                        std::vector<cv::DMatch>& filtered_matches,
                                        std::vector<cv::KeyPoint>& keys_A, 
                                        std::vector<cv::KeyPoint>& keys_B){
    std::vector<cv::DMatch> good_matches;
    cv::Point2d p_A, p_B;
    cv::Point2d p_cntr = cv::Point2d(next_.frame.cols/2, next_.frame.rows/2);
    double dist_interpoint, dist_center;
    for(int i = 0; i < matches.size(); i++){
        p_A = keys_A[matches[i].queryIdx].pt;
        p_B = keys_B[matches[i].trainIdx].pt;

        dist_interpoint = sqrt( pow((p_A.x-p_B.x), 2) + pow((p_A.y-p_B.y), 2) );
        dist_center = sqrt( pow((p_A.x-p_cntr.x), 2) + pow((p_A.y-p_cntr.y), 2) );
        if(dist_interpoint < filter_distance_*dist_center/p_cntr.x)
            good_matches.push_back(matches[i]);
    }
    filtered_matches = good_matches; 
}

void PoseEstimator::ConvertPoints3D(const std::vector<cv::DMatch>& matches,
                                    const std::vector<cv::KeyPoint>& keys,
                                    Eigen::MatrixXd& points_3d){
    points_3d = Eigen::MatrixXd(matches.size(), 3); 
    for(int i = 0; i < matches.size(); i++){
        points_3d.row(i) << keys[matches[i].queryIdx].pt.x,
                            keys[matches[i].queryIdx].pt.y,
                            1;
    }
}

double PoseEstimator::CalcRT_RANSAC(const Eigen::MatrixXd& points_3d_A, 
                                    const Eigen::MatrixXd& points_3d_B, 
                                    Eigen::Matrix3d& rot_mat,
                                    Eigen::Vector3d& t_vec){
    std::vector<int> filtered_indexes;
    Eigen::MatrixXd points_3d_A_sample = Eigen::MatrixXd(pR_.sample_size, 3); 
    Eigen::MatrixXd points_3d_B_sample = Eigen::MatrixXd(pR_.sample_size, 3); 
    Eigen::Matrix3d rot_mat_hpths;
    Eigen::Vector3d t_vec_hpths;
    double error; 
    int rand_index; 
    for(int iter = 0; iter < pR_.iterations; iter++){
        std::vector<int> sample_indexes; 
        srand(time(0));
        for(int i = 0; i < pR_.sample_size; i++){
            rand_index = rand() % points_3d_A.rows(); 
            if(std::find(sample_indexes.begin(), sample_indexes.end(), rand_index) != sample_indexes.end()){
                i--; 
                continue;
            }
            sample_indexes.push_back(rand_index); 

            points_3d_A_sample.row(i) << points_3d_A.row(rand_index); 
            points_3d_B_sample.row(i) << points_3d_B.row(rand_index); 
        }

        FitLSR(points_3d_A_sample, points_3d_B_sample, rot_mat_hpths, t_vec_hpths); 

        std::vector<int> good_indexes; 
        for(int i = 0; i < points_3d_A.rows(); i++){
            error = (rot_mat_hpths*points_3d_A.row(i).transpose() + t_vec_hpths - points_3d_B.row(i).transpose()).norm();
            if(error < pR_.threshold){
                good_indexes.push_back(i); 
            }
        }

        if(good_indexes.size()-pR_.sample_size > filtered_indexes.size()){
            filtered_indexes = good_indexes; 
        }

        if(filtered_indexes.size() > pR_.min_score)
            break; 
    }
    if(filtered_indexes.size() > 3){
        Eigen::MatrixXd points_3d_A_filtered = Eigen::MatrixXd(filtered_indexes.size(), 3); 
        Eigen::MatrixXd points_3d_B_filtered = Eigen::MatrixXd(filtered_indexes.size(), 3);
        for(int i = 0; i < filtered_indexes.size(); i++){
            points_3d_A_filtered.row(i) << points_3d_A.row(filtered_indexes[i]); 
            points_3d_B_filtered.row(i) << points_3d_B.row(filtered_indexes[i]); 
        }
        FitLSR(points_3d_A_filtered, points_3d_B_filtered, rot_mat, t_vec); 
        return 100*filtered_indexes.size()/points_3d_A.rows(); 
    }else{
        return 100*filtered_indexes.size()/points_3d_A.rows(); 
    }
}

void PoseEstimator::FitLSR(Eigen::MatrixXd& points_3d_A, 
                           Eigen::MatrixXd& points_3d_B,
                           Eigen::Matrix3d& rot_mat,
                           Eigen::Vector3d& t_vec){
    Eigen::Vector3d centroid_A, centroid_B;

    CalcCentroid(points_3d_A, centroid_A); 
    CalcCentroid(points_3d_B, centroid_B);
    
    CenterSet(centroid_A, points_3d_A); 
    CenterSet(centroid_B, points_3d_B);    

    ExtractRotationA2B(points_3d_A, points_3d_B, rot_mat);
    CheckRCase(rot_mat); 
    ExtractTranslationA2B(rot_mat, centroid_A, centroid_B, t_vec);
}

void PoseEstimator::CalcCentroid(const Eigen::MatrixXd& points_3d, Eigen::Vector3d& centroid){
    centroid = points_3d.colwise().mean();
}

void PoseEstimator::CenterSet(const Eigen::Vector3d& centroid, Eigen::MatrixXd& points_3d){
    points_3d = points_3d.rowwise() - centroid.transpose(); 
}

void PoseEstimator::ExtractRotationA2B(const Eigen::MatrixXd& points_3d_centered_A,
                                       const Eigen::MatrixXd& points_3d_centered_B,
                                       Eigen::Matrix3d& rot_mat){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(points_3d_centered_A.transpose()*points_3d_centered_B, 
                                          Eigen::ComputeFullV | Eigen::ComputeFullU);
    rot_mat = svd.matrixV()*svd.matrixU().transpose(); 
}

void PoseEstimator::ExtractTranslationA2B(const Eigen::Matrix3d& rot_mat,
                                          const Eigen::Vector3d& centroid_A,
                                          const Eigen::Vector3d& centroid_B, 
                                          Eigen::Vector3d& t_vec){
    t_vec = centroid_B - rot_mat*centroid_A; 
}

void PoseEstimator::CheckRCase(Eigen::Matrix3d& rot_mat){
    if(rot_mat.determinant() < 0){ 
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(rot_mat, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::MatrixXd Vt = svd.matrixV(); 
        Eigen::Vector3d M = Eigen::Vector3d(1, 1, -1);
        Eigen::Matrix3Xd C = Vt.array().colwise() * M.array();
        rot_mat =  Vt * svd.matrixU().transpose();
    }
} 

void PoseEstimator::ConvertR2YPR(const Eigen::Matrix3d& rot_mat, Eigen::Vector3d& angles){
    angles = rot_mat.eulerAngles(2, 1, 0) * 360/CV_PI; 
    for(int i = 0; i < angles.size(); i++)
        if(angles[i] > 180)
            angles[i] -= 360; 
}

void PoseEstimator::ConvertR2Q(const Eigen::Matrix3d& rot_mat, Eigen::Quaterniond& quat){
    quat = rot_mat; 
}

void PoseEstimator::CycleData(){
    prev_ = next_; 
}

void PoseEstimator::HandleError(){
    switch(error_){
        case NO_ERRORS:                     error_message_ = "No errors"; break; 
        case NO_MATCHES_FOUND:              error_message_ = "No matches between images"; break;
        case NOT_ENOUGH_MATCHES:            error_message_ = "Not enough matches"; break;
        case NO_GOOD_SOLUTION_RANSAC:       error_message_ = "No good solution found"; break;
        default:                            error_message_ = "Something is broken"; 
    }
}