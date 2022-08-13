#include "tile_depthmeter_class.hpp"

using namespace std; 
using namespace cv;

TileDepthmeter::TileDepthmeter() {
}

void TileDepthmeter::SetTileSize(float heigth, float width) { 
    tile_vals.height = heigth;
    tile_vals.width = width;
}

void TileDepthmeter::SetCameraIntrinsics(vector<vector<float>> intr_vals) {
    camera_vals.fx = intr_vals[0][0];
    camera_vals.fy = intr_vals[1][1];
    camera_vals.cx = intr_vals[0][2];
    camera_vals.cy = intr_vals[1][2];

}

void TileDepthmeter::SetFilterParams(int sample_size, int cycles, float thrshld, int min_inliers) {
    filter_vals.sample_size = sample_size; 
    filter_vals.iterations = cycles;
    filter_vals.threshold = thrshld;
    filter_vals.min_score = min_inliers;
}

void TileDepthmeter::SetPreprocessingParams(Size kernel, int laplace_krnl, int bin_thrshld, float thrshld_1, float thrshld_2) {
    pp_vals.gaussian_kernel = kernel;
    pp_vals.laplacian_kernel = laplace_krnl;
    pp_vals.bin_threshold = bin_thrshld;
    pp_vals.canny_threshold_1 = thrshld_1;
    pp_vals.canny_threshold_2 = thrshld_2;
}

void TileDepthmeter::SetFrame(Mat new_frame) {
    frame = new_frame; 
    pipeline.frame_raw = frame;
}

float TileDepthmeter::CalcDistance() {
    scale_factor_ = CalcScale();
    float distance = camera_vals.fx*scale_factor_;
    return distance;
}

float TileDepthmeter::CalcScale() {
    vector<RotatedRect> good_rects = FindRectangles();

    float area_perceived = FindDominantArea(good_rects);
    float area_real = tile_vals.height*tile_vals.width;

    scale_factor_ = sqrt(area_real / area_perceived);
    return scale_factor_; 
}

void TileDepthmeter::PreprocessImage() {
    GaussianBlur(frame, frame, pp_vals.gaussian_kernel, 0, 0);
    pipeline.frame_blurred_1 = frame;

    cvtColor(frame, frame, COLOR_BGR2GRAY);
    pipeline.frame_gray = frame;

    Laplacian(frame, frame, CV_8U, pp_vals.laplacian_kernel, 1, 0);
    pipeline.frame_laplacian = frame; 

    GaussianBlur(frame, frame, pp_vals.gaussian_kernel, 0, 0);
    pipeline.frame_blurred_2 = frame; 

    threshold(frame, frame, pp_vals.bin_threshold, 255, THRESH_BINARY_INV);
    pipeline.frame_bin = frame; 

    Canny(frame, frame, pp_vals.canny_threshold_1, pp_vals.canny_threshold_2);
    pipeline.frame_edges = frame; 
}

vector<RotatedRect> TileDepthmeter::FindRectangles() {
    vector<vector<Point> > contours;
    findContours(frame, contours, RETR_LIST, CHAIN_APPROX_NONE);

    vector<RotatedRect> minimal_rectangles(contours.size());
    int rect_cntr = 0; 
    for(int i = 0; i < contours.size(); i++) {
        if(minAreaRect(contours[i]).size.area() > 10) {
            minimal_rectangles[rect_cntr] = minAreaRect(contours[i]);
            rect_cntr++;
        }
    }
    minimal_rectangles.erase(minimal_rectangles.begin() + rect_cntr, minimal_rectangles.end());
    return minimal_rectangles; 
}

float TileDepthmeter::FindDominantArea(vector<RotatedRect> minimal_rectangles) {
    int score = 0; 
    float area = 0;
    if(minimal_rectangles.size() > 100)
        for(int iter = 0; iter < filter_vals.iterations; iter++) {
            float rect_area = 0; 

            vector<int> indexes(filter_vals.sample_size, 0);
            srand(time(0));
            for(int i = 0; i < filter_vals.sample_size; i++) {
                indexes[i] = rand()%minimal_rectangles.size();
            }

            int points_cntr = 0;
            int idx_cntr = 0;
            for(int i = 0; i < minimal_rectangles.size(); i++) {
                if(i == indexes[idx_cntr]) {
                    idx_cntr++;
                    continue;
                }
                if(minimal_rectangles[i].size.area() - rect_area < filter_vals.threshold) {
                    points_cntr++; 
                } 
            }
            if(points_cntr > score  && rect_area > area) {
                score = points_cntr; 
                area = rect_area; 
            }
        }
    else
        for(int iter = 0; iter < minimal_rectangles.size(); iter++) {
            float rect_area = 0; 

            rect_area = minimal_rectangles[iter].size.area(); 
            
            int points_cntr = 0;
            for(int i = 0; i < minimal_rectangles.size(); i++) {
                if(i == iter) {
                    continue;
                }
                if(minimal_rectangles[i].size.area() - rect_area < filter_vals.threshold) {
                    points_cntr++; 
                } 
            }
            if(points_cntr > score  && rect_area > area) {
                score = points_cntr; 
                area = rect_area; 
            }
        }

    return area; 
}
