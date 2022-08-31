// #include "pose_estimator_class.hpp"

// using namespace std;

// void PoseEstimator::SetPrevFrame(Mat frame){
//     frame_next = frame; 
//     SharpenImage();
//     CycleData(); 
// }

// void PoseEstimator::SetNextFrame(Mat frame){
//     frame_next = frame; 
//     SharpenImage();
// }

// void PoseEstimator::SetGaussParams(int stype, int dtype, Size k, float s_x, float s_y){
//     pGau.src_type = stype; 
//     pGau.dst_type = dtype; 
//     pGau.ksize = k;
//     pGau.sigma_x = s_x;
//     pGau.sigma_y = s_y; 
// }

// void PoseEstimator::SetLaplaceParams(int stype, int dtype, int k){
//     pLap.src_type = stype; 
//     pLap.dst_type = dtype; 
//     pLap.ksize = k; 
    
// }

// void PoseEstimator::SetMatcherParams(int number_of_cells, int border_size, int max_spread){
//     pMat.number_of_cells = number_of_cells; 
//     pMat.border_size = border_size; 
//     pMat.number_of_cells = number_of_cells;
// }

// Mat PoseEstimator::GetProcessedFrame(){
//     Mat frame; 
//     gpu_frame_next.download(frame);
//     return frame; 
// }

// vector<double> PoseEstimator::GetYPR(){
//     vector<double> temp{ YPR[0],
//                         YPR[1],
//                         YPR[2]};
//     return temp; 
// }

// vector<double> PoseEstimator::GetQ(){
//     vector<double> temp{Q.w(),
//                         Q.x(),
//                         Q.y(),
//                         Q.z()};
//     return temp; 
// }

// vector<double> PoseEstimator::GetT(){
//     vector<double> temp{T[0],
//                         T[1],
//                         T[2]};
//     return temp; 
// }

// bool PoseEstimator::CalculateMotion(){
//     ExtractMotion(); 
//     FilterMoves(); 

//     if(deltas_filtered.rows() < 3){
//         CycleData();
//         return false; 
//     }

//     Convert3D();
//     CalcNextCentroid();
//     CenterSet();
//     ExtractRotation();
//     CheckRCase(); 
//     Convert2YPR(); 
//     Convert2Q(); 
//     CalculateTranslation();

//     CycleData();
//     return true; 
// }

// void PoseEstimator::SharpenImage(){
//     gpu_frame_next.upload(frame_next); 
//     gpu_frame_mask.upload(frame_next); 

//     cuda::cvtColor(gpu_frame_next, gpu_frame_next, COLOR_BGR2GRAY);
//     cuda::cvtColor(gpu_frame_mask, gpu_frame_mask, COLOR_BGR2GRAY);

//     gpu_frame_next.convertTo(gpu_frame_next, CV_8U);
//     gpu_frame_mask.convertTo(gpu_frame_mask, CV_8U);

//     cuda::normalize(gpu_frame_next, gpu_frame_next, 0, pow(2, 8), NORM_MINMAX, CV_8U);
//     cuda::normalize(gpu_frame_mask, gpu_frame_mask, 0, pow(2, 8), NORM_MINMAX, CV_8U);

//     gaussian_filter->apply(gpu_frame_mask, gpu_frame_mask);
//     laplacian_filter->apply(gpu_frame_mask, gpu_frame_mask); 

//     cuda::subtract(gpu_frame_next, gpu_frame_mask, gpu_frame_next);
// }

// void FilterMoves(){

// }

// void PoseEstimator::Convert3D(){
//     points3d_prev = MatrixXd(matches_filtered.size(), 3); 
//     points3d_next = MatrixXd(matches_filtered.size(), 3); 
//     for(int i = 0; i < matches_filtered.size(); i++){
//         points3d_prev.row(i) << keys_prev[matches_filtered[i].queryIdx].pt.x,
//                                 keys_prev[matches_filtered[i].queryIdx].pt.y,
//                                 1;

//         points3d_next.row(i) << keys_next[matches_filtered[i].trainIdx].pt.x,
//                                 keys_next[matches_filtered[i].trainIdx].pt.y,
//                                 1;
//     }
// }

// void PoseEstimator::CalcNextCentroid(){
//     centroid_prev = points3d_prev.colwise().mean();
//     centroid_next = points3d_next.colwise().mean(); 
// }

// void PoseEstimator::CenterSet(){
//     points3d_prev = (-points3d_prev).rowwise() + centroid_prev.transpose(); 
//     points3d_next = (-points3d_next).rowwise() + centroid_next.transpose(); 
// }

// void PoseEstimator::ExtractRotation(){
//     H = points3d_prev.transpose() * points3d_next; 

//     JacobiSVD<MatrixXd> svd(H, ComputeFullV | ComputeFullU);

//     R = svd.matrixV() * svd.matrixU().transpose(); 
// }

// void PoseEstimator::CheckRCase(){
//     if(R.determinant() < 0){ 
//         JacobiSVD<MatrixXd> svd(R, ComputeFullV | ComputeFullU);
//         MatrixXd Vt = svd.matrixV(); 
//         Vector3d M = Vector3d(1, 1, -1);
//         Matrix3Xd C = Vt.array().colwise() * M.array();
//         R =  Vt * svd.matrixU().transpose();
//     }
// } 

// void PoseEstimator::Convert2YPR(){
//         YPR = R.eulerAngles(2, 1, 0) * 360/CV_PI; 
//         for(int i = 0; i < YPR.size(); i++)
//             if(YPR[i] > 180)
//                 YPR[i] -= 360; 
// }

// void PoseEstimator::Convert2Q(){
//     Q = R; 
// }

// void PoseEstimator::CalculateTranslation(){
//     T = centroid_next - R*centroid_prev; 
// }

// void PoseEstimator::CycleData(){
//     frame_prev = frame_next;
//     gpu_frame_prev = gpu_frame_next;
