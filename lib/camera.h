#ifndef CAMERA_H
#define CAMERA_H

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <stdio.h>
#include <iostream>


class Camera
{
public:
    Camera(const float _f_x, const float _f_y, const float _c_x, const float _c_y);
    Camera(float const *_f, float const *_c, float *_dist);

    Eigen::Vector2f Project(Eigen::Vector3f _X);
    Eigen::Vector3f Unproject(Eigen::Vector2f _uv, float _inv_depth);
    Eigen::Vector4f UnprojectInc(Eigen::Vector2f _uv, float _inv_depth);

    // Same methods but taking into account distortion
    Eigen::Vector2f ProjectDist(Eigen::Vector3f _X);
    Eigen::Vector3f UnprojectDist(Eigen::Vector2f _uv, float _inv_depth);
    Eigen::Vector4f UnprojectDistInc(Eigen::Vector2f _uv, float _inv_depth);

    Eigen::Vector2f Distort(Eigen::Vector2f _u_corr);
    Eigen::Vector2f Undistort(Eigen::Vector2f _u_dist);

    cv::Mat GetK();
    cv::Mat GetDist();

private:
    const float f_x;
    const float f_x_1; // 1/f_x
    const float f_y;
    const float f_y_1; // 1/f_y
    cv::Mat cv_K; // camera matrix

    const float c_x;
    const float c_y;
    const float dist[4]; // k1, k2, p1, p2
    cv::Mat cv_dist;
    Eigen::Matrix3f K;
};

#endif
