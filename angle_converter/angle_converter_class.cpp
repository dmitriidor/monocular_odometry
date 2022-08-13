#include "angle_converter_class.hpp"

using namespace std; 
using namespace Eigen;

AngleConverter::AngleConverter() {
}

void AngleConverter::SetAndConvertR(vector< vector<float> >rot_mat) { 
    rot_mat_std = rot_mat; 
    rot_mat_eigen << rot_mat[0][0], rot_mat[0][1], rot_mat[0][2],
                     rot_mat[1][0], rot_mat[1][1], rot_mat[1][2],
                     rot_mat[2][0], rot_mat[2][1], rot_mat[2][2];

    ConvertR2Q();
    ConvertR2YPR();
}

void AngleConverter::SetAndConvertQ(vector<float> quaternion) {
    q_std = quaternion; 
    q_eigen = Map<Quaternionf>(quaternion.data());

    ConvertQ2R();
    ConvertQ2YPR();
}

void AngleConverter::SetAndConvertYPR(vector<float> yaw_pitch_roll) {
    ypr_std = yaw_pitch_roll;
    ypr_eigen = Map<Vector3f>(yaw_pitch_roll.data());

    ConvertYPR2Q();
    ConvertYPR2R();
}

vector< vector<float> > AngleConverter::GetRotMatSTD() {
    return rot_mat_std;
}

vector<float> AngleConverter::GetQuaternionSTD() {
    return q_std;
}

vector<float> AngleConverter::GetYawPitchRollSTD() {
    return ypr_std;
}

Matrix3f AngleConverter::GetRotMatEigen() {
    return rot_mat_eigen;
}

Quaternionf AngleConverter::GetQuaternionEigen(){
    return q_eigen;
}

Vector3f AngleConverter::GetYawPitchRollEigen() {
    return ypr_eigen; 
}

void AngleConverter::ConvertR2Q(){
    q_eigen = rot_mat_eigen; 
    q_std = vector<float>(q_eigen.coeffs().data(), q_eigen.coeffs().data() + 
        q_eigen.coeffs().rows()*q_eigen.coeffs().cols());
}

void AngleConverter::ConvertR2YPR() {
    ypr_eigen = rot_mat_eigen.eulerAngles(2, 1, 0); 
    ypr_std = vector<float>(ypr_eigen.data(), ypr_eigen.data() + ypr_eigen.rows()*ypr_eigen.cols());
}

void AngleConverter::ConvertQ2R() {
    rot_mat_eigen = q_eigen.toRotationMatrix();
    rot_mat_std = {vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()),
                   vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()*2),
                   vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()*3)};
}

void AngleConverter::ConvertQ2YPR() {
    ypr_eigen = q_eigen.toRotationMatrix().eulerAngles(0, 1, 2);
    ypr_std = vector<float>(ypr_eigen.data(), ypr_eigen.data() + 
        ypr_eigen.rows()*ypr_eigen.cols());
}

void AngleConverter::ConvertYPR2Q(){
    q_eigen = AngleAxisf(ypr_std[0], Vector3f::UnitX())
            * AngleAxisf(ypr_std[1], Vector3f::UnitY())
            * AngleAxisf(ypr_std[2], Vector3f::UnitZ());
    q_std = vector<float>(q_eigen.coeffs().data(), q_eigen.coeffs().data() + 
                            q_eigen.coeffs().rows()*q_eigen.coeffs().cols());;
}

void AngleConverter::ConvertYPR2R() {
    rot_mat_eigen = q_eigen.toRotationMatrix(); 
    rot_mat_std = {vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()),
                vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()*2),
                vector<float>(rot_mat_eigen.data(), rot_mat_eigen.data() + rot_mat_eigen.rows()*3)};
}