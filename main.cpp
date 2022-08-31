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

    gpu_frame_1 = gpu_frame_2; 

    Ptr<cuda::TemplateMatching> matcher = cuda::createTemplateMatching(CV_8U, TM_CCOEFF, Size(0, 0));

    cuda::GpuMat gpu_matching_results; 
    Mat matching_results;
    
    int shift = (frame_1.cols - frame_1.rows)/2; 
    Rect square_image(shift, 0, frame_1.rows, frame_1.rows);

    int number_of_cells = 16;
    int cell_size = frame_1.rows/sqrt(number_of_cells);

    int border_size = 10; 


    int rows = sqrt(number_of_cells); 
    int cols = sqrt(number_of_cells);
    MatrixXi good_cells; 
    if(number_of_cells%2 == 0){
        good_cells = MatrixXi(4*2, 2);
        good_cells <<   0,         cols/2-1, 
                        0,         cols/2, 
                        rows/2-1,  0, 
                        rows/2-1,  cols-1, 
                        rows/2,    0, 
                        rows/2,    cols-1,
                        rows-1,    cols/2-1, 
                        rows-1,    cols/2;
    }else{
        good_cells = MatrixXi(4, 2);
        good_cells << 0,             (int)cols/2, 
                      (int)rows/2,   0, 
                      (int)rows/2,   cols-1, 
                      rows-1,        (int)cols/2;
    }

    double tang = 0;
         

    Rect subregion_for_matching(border_size, border_size, cell_size - 2*border_size, cell_size - 2*border_size);
    cuda::GpuMat gpu_subregion_1, gpu_subregion_2; 
    cuda::GpuMat gpu_subregion_to_match, gpu_region_for_matching; 

    int x_new, y_new; 
    float dx_avg = 0, dy_avg = 0; 


    gpu_frame_2 = gpu_frame_1(square_image);

    MatrixXd deltas(good_cells.rows(), 2);
    MatrixXd positions(good_cells.rows(), 2); 
    MatrixXd deltas_filtered; 
    float max_distance = 1; 

    PoseEstimator estimator; 
    estimator.SetPrevFrame(frame_1); 

    clock_t start, end; 
    for(int iter = 1; iter < 1000000+1; iter++) {
        cap >> frame_2;

        estimator.SetNextFrame(frame_2);
        if(estimator.CalculateMotion())
            cout << "Hmmm: " << estimator.GetYPR()[0] << endl;
        else
            cout << "Doh!" << endl; 

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
        cuda::minMax(gpu_frame_2, &minimum_value, &maximum_value); 
        gpu_frame_2.convertTo(gpu_frame_2, CV_8U, 1/pow(2, 8));

        gpu_frame_2 = gpu_frame_2(square_image); 
        dx_avg = 0; 
        dy_avg = 0;

        int good_cell_counter = 0; 
        for(int i = 0; i < sqrt(number_of_cells); i++){
            for(int j = 0; j < sqrt(number_of_cells); j++){
                if(good_cells.row(good_cell_counter)[0] == i &&  good_cells.row(good_cell_counter)[1] == j){
                    good_cell_counter++;
                }else{
                    continue;
                }
                int subregion_shift_x = border_size + cell_size * i;
                int subregion_shift_y = border_size + cell_size * j;
                Rect subregion_to_match(subregion_shift_x, subregion_shift_y, cell_size - 2*border_size, cell_size - 2*border_size); 
                gpu_subregion_to_match = gpu_frame_1(subregion_to_match);

                int region_shift_x = cell_size * i;
                int region_shift_y = cell_size * j;
                Rect region_for_matching(region_shift_x, region_shift_y, cell_size, cell_size);
                gpu_region_for_matching = gpu_frame_2(region_for_matching);
                matcher->match(gpu_region_for_matching, gpu_subregion_to_match, gpu_matching_results);
                
                gpu_matching_results.download(matching_results); 
                double minVal, maxVal;
                Point minLoc, maxLoc;
                minMaxLoc(matching_results, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

                x_new = maxLoc.x - border_size;
                y_new = maxLoc.y - border_size; 
                deltas.row(good_cell_counter-1) << x_new, y_new; 
                positions.row(good_cell_counter-1) << subregion_shift_x, subregion_shift_y; 


                // cout << "dx:" << x_new - border_size << " dy:" << y_new - border_size << endl;

                dx_avg += x_new; 
                dy_avg += y_new; 


                gpu_frame_1.download(frame_1);
                gpu_frame_2.download(frame_2);
                rectangle(frame_2, maxLoc + Point(region_shift_x, region_shift_y), 
                                Point(maxLoc.x + gpu_subregion_to_match.cols +region_shift_x, maxLoc.y + gpu_subregion_to_match.rows+region_shift_y), Scalar::all(0), 2, 8, 0 );
                rectangle(frame_2, region_for_matching, Scalar::all(0), 2, 8, 0);
                rectangle(frame_2, subregion_to_match, Scalar::all(0), 2, 8, 0);
                rectangle(frame_1, region_for_matching, Scalar::all(0), 2, 8, 0);
                rectangle(frame_1, subregion_to_match, Scalar::all(0), 2, 8, 0);
                imshow("Matching result", frame_2);
                imshow("1st frame", frame_1);
                waitKey(0); 

                if(good_cell_counter == good_cells.rows()){
                    goto loop_exit; 
                }
            }
        }
        
        loop_exit:

        int best_score = 0; 
        float max_distance = 5; 
        MatrixXd deltas_filtered(0, 2); 
        MatrixXd position_filtered(0, 2); 
        for(int i = 0; i < deltas.rows(); i++){
            for(int j = i+1; j < deltas.rows(); j++){
                Vector2d centroid; 
                centroid << 0, 0; 
                centroid = (deltas.row(i) + deltas.row(j))/2;

                int score = 0;
                MatrixXd deltas_inliers(0, 2);
                MatrixXd positions_inliers(0, 2);
                for(int k = 0; k < deltas.rows(); k++){
                    if((deltas.row(k)-centroid.transpose()).norm() < max_distance){
                        score++; 
                        deltas_inliers.conservativeResize(deltas_inliers.rows()+1, NoChange);
                        deltas_inliers.row(deltas_inliers.rows()-1) = deltas.row(k); 

                        positions_inliers.conservativeResize(positions_inliers.rows()+1, NoChange);
                        positions_inliers.row(positions_inliers.rows()-1) = positions.row(k); 
                    }

                    if(score > best_score){
                        deltas_filtered = deltas_inliers; 
                        position_filtered = positions_inliers; 
                    }
                }
            }
        }

        MatrixXd pnts3d_1_eigen(deltas_filtered.rows(), 3); 
        MatrixXd pnts3d_2_eigen(deltas_filtered.rows(), 3); 
        for(int i = 0; i < deltas_filtered.rows(); i++){
            pnts3d_2_eigen.row(i) <<    deltas_filtered.row(i)[0] + position_filtered.row(i)[0],
                                        deltas_filtered.row(i)[1] + position_filtered.row(i)[1],
                                        1; 
            pnts3d_1_eigen.row(i) <<    position_filtered.row(i)[0],
                                        position_filtered.row(i)[1],
                                        1; 
        }

        Vector3d centroid_1, centroid_2;

        centroid_1 <<   pnts3d_1_eigen.col(0).mean(), 
                        pnts3d_1_eigen.col(1).mean(),
                        pnts3d_1_eigen.col(2).mean(); 
        
        centroid_2 <<   pnts3d_2_eigen.col(0).mean(), 
                        pnts3d_2_eigen.col(1).mean(),
                        pnts3d_2_eigen.col(2).mean(); 

        for(int i = 0; i < pnts3d_2_eigen.rows(); i++){
            pnts3d_2_eigen.row(i) -= centroid_2; 
            pnts3d_1_eigen.row(i) -= centroid_1;
        }
        

        if(pnts3d_2_eigen.rows() < 3)
            continue;

        MatrixXd H = pnts3d_1_eigen.transpose() * pnts3d_2_eigen; 

        cout << "\nH: \n" << H << endl;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullV | Eigen::ComputeFullU);

        Matrix3d R = svd.matrixV() * svd.matrixU().transpose(); 

        if(R.determinant() < 0) { 
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
            MatrixXd Vt = svd.matrixV(); 
            Eigen::Vector3d M = Eigen::Vector3d(1, 1, -1);
            Eigen::Matrix3d C = Vt.array().colwise() * M.array();
            R =  Vt * svd.matrixU().transpose();
        }

        Vector3d ang = R.eulerAngles(2, 1, 0); 
        for(int i = 0; i < ang.size(); i++){
            ang[i] *= 360/CV_PI; 
            if(ang[i] > 180){
                ang[i] -= 360;
            }
        }
        tang += ang[0];
        for(int i = 0; i < ang.size(); i++)
            cout << "Angle "<< i << ": " << ang[i] << endl; 
        cout << "Total angle: " << tang << endl; 

        Vector3d T = centroid_2 - R*centroid_1;
        cout << "Translation: " << T << endl; 


        // Vector3f centroid_1_eigen;
        // centroid_1_eigen << centroid_1.x, centroid_1.y, 1;
        // Vector3f centroid_2_eigen;
        // centroid_2_eigen << centroid_2.x, centroid_2.y, 1; 

        // Vector3f T = centroid_2_eigen - centroid_1_eigen;

        // cout << "Averages: " << deltas_filtered.col(0).mean() << " " << deltas_filtered.col(1).mean() << endl; 
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
