#ifndef MATHS_H
#define MATHS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>

class Maths
{
public:
    static Eigen::Matrix<float,3,3> Skew(Eigen::Vector3f vec);
    static Eigen::Matrix<float,3,3> Skew(float phi1, float phi2, float phi3);
    static Eigen::Vector3f Unskew(Eigen::Matrix<float,3,3> mat);

    static Eigen::Matrix<float,3,3> ExpSO3(Eigen::Vector3f phi);
    static Eigen::Matrix<float,3,3> ExpSO3(double phi1, double phi2, double phi3);
    static Eigen::Vector3f LogSO3(Eigen::Matrix<float,3,3> R);

    static Eigen::Matrix<float,4,4> ExpSE3(Eigen::Vector3f phi);
    static Eigen::Matrix<float,6,1> LogSE3(Eigen::Matrix<float,4,4> T);
    static Eigen::Matrix<float,4,4> InvSE3(Eigen::Matrix<float,4,4> T);

    // Conversions
    static Eigen::MatrixXf cvMat2Eigmat(cv::Mat cvM);
    static cv::Mat Eigmat2Cvmat(Eigen::MatrixXf eigM);
    static Eigen::Vector2f cvPoint2EigVec2(cv::Point2f cvP);
    static Eigen::Vector3f cvPoint2EigVec3(cv::Point3f cvP);
    static Eigen::Vector2f cvMat2EigVec2(cv::Mat cvMat);
    static Eigen::Vector3f cvMat2EigVec3(cv::Mat cvMat);
    static cv::Point2f EigV2f2cvPt2(Eigen::Vector2f _v_eig);

    // Geometry
    static void MotFromEss(const cv::Mat &_E, cv::Mat &_R1, cv::Mat &_R2, cv::Mat &_t);

};

#endif // MATHS_H

