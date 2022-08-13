#ifndef TileDepthmeterClass_H
#define TileDepthmeterClass_H

#include <vector>
#include <opencv4/opencv2/opencv.hpp>

using namespace std; 
using namespace cv;

class TileDepthmeter{
    public:
    TileDepthmeter(); 
    void SetTileSize(float heigth, float width);
    void SetCameraIntrinsics(vector<vector<float>> intr_vals);
    void SetFilterParams(int sample_size, int cycles, float thrshld, int min_inliers);
    void SetPreprocessingParams(Size kernel, int laplace_krnl, int bin_thrshld, float thrshld_1, float thrshld_2);
    void SetFrame(Mat new_frame); 

    void PreprocessImage();
    float CalcDistance(); 
    float CalcScale(); 

    struct FramePipeline { 
        Mat frame_raw; 
        Mat frame_blurred_1;
        Mat frame_gray; 
        Mat frame_laplacian; 
        Mat frame_blurred_2; 
        Mat frame_bin; 
        Mat frame_edges; 
    };
    FramePipeline pipeline;
    
    private:
    vector<RotatedRect> FindRectangles();
    float FindDominantArea(vector<RotatedRect> minimal_rectangles); 

    struct TileParams {
        float height = 0;
        float width = 0; 
        float area = height*width; 
    };
    TileParams tile_vals;

    struct CameraParams {
        float fx = 0;
        float fy = 0;
        float cx = 0;
        float cy = 0;
    };
    CameraParams camera_vals;

    Mat frame; 

    struct PreprocessingParams {
        Size gaussian_kernel;
        int bin_threshold;
        int laplacian_kernel;
        int canny_threshold_1;
        int canny_threshold_2;
    };
    PreprocessingParams pp_vals; 

    struct FilterParams {
        int sample_size = 0;
        int iterations = 0; 
        int threshold = 0; 
        int min_score = 0;
    };
    FilterParams filter_vals;

    float scale_factor_ = 0; 
};

#endif