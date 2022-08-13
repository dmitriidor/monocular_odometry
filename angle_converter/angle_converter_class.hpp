#ifndef AngleConverterClass_H
#define AngleConverterClass_H

#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <algorithm>

using namespace std; 
using namespace Eigen; 

class AngleConverter{
    public:
    AngleConverter(); 
    void SetAndConvertR(vector< vector<float> >rot_mat);
    void SetAndConvertQ(vector<float> quaternion);
    void SetAndConvertYPR(vector<float> yaw_pitch_roll);

    vector< vector<float> > GetRotMatSTD();
    vector<float> GetQuaternionSTD();
    vector<float> GetYawPitchRollSTD();

    Matrix3f GetRotMatEigen();
    Quaternionf GetQuaternionEigen();
    Vector3f GetYawPitchRollEigen();

    private:
    void ConvertR2Q();
    void ConvertR2YPR();

    void ConvertQ2R();
    void ConvertQ2YPR();

    void ConvertYPR2Q();
    void ConvertYPR2R();

    vector< vector<float> >rot_mat_std {{0,0,0},
                                        {0,0,0},
                                        {0,0,0}};
    Matrix3f rot_mat_eigen;

    vector<float> q_std {0, 0, 0, 0}; 
    Quaternionf q_eigen; 

    vector<float> ypr_std {0, 0, 0};
    Vector3f ypr_eigen {0, 0, 0}; 
};

#endif