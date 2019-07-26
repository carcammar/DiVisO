#include "camera.h"

Camera::Camera(const float _f_x, const float _f_y, const float _c_x, const float _c_y):
    f_x(_f_x), f_x_1(1.f/_f_x), f_y(_f_y), f_y_1(1.f/_f_y), c_x(_c_x), c_y(_c_y), dist{0.f,0.f,0.f,0.f}
{
    cv_K = cv::Mat(3,3,CV_32F);
    cv_K.at<float>(0,0)=f_x;
    cv_K.at<float>(1,1)=f_y;
    cv_K.at<float>(0,2)=c_x;
    cv_K.at<float>(1,2)=c_y;
    cv_K.at<float>(2,2)=1.f;
    cv_dist = cv::Mat(4, 1, CV_32F, 0.0);
    K = Maths::cvMat2Eigmat(cv_K);
}

Camera::Camera(float const *_f, float const *_c, float *_dist):
    f_x(_f[0]), f_x_1(1.f/_f[0]), f_y(_f[1]), f_y_1(1.f/_f[1]), c_x(_c[0]), c_y(_c[1]),
    dist{_dist[0], _dist[1], _dist[2], _dist[3]}
{
    cv_K = cv::Mat(3,3,CV_32F);
    cv_K.at<float>(0,0)=f_x;
    cv_K.at<float>(1,1)=f_y;
    cv_K.at<float>(0,2)=c_x;
    cv_K.at<float>(1,2)=c_y;
    cv_K.at<float>(2,2)=1.f;
    cv_dist = cv::Mat(4, 1, CV_32F, _dist);
    K = Maths::cvMat2Eigmat(cv_K);
}

Eigen::Vector2f Camera::Distort(Eigen::Vector2f _u_corr)
{

}

Eigen::Vector2f Camera::Undistort(Eigen::Vector2f _u_dist)
{

}

Eigen::Vector2f Camera::Project(Eigen::Vector3f _X)
{
    Eigen::Vector2f uv;
    float z_1 = 1/_X[2];
    uv[0] = z_1*_X[0]*f_x+c_x;
    uv[1] = z_1*_X[1]*f_y+c_y;
    return uv;
}

Eigen::Vector3f Camera::Unproject(Eigen::Vector2f _uv, float _inv_depth)
{
    Eigen::Vector3f c_X;
    float z = 1.f/_inv_depth;
    c_X[0]=(_uv[0]-c_x)*f_x_1*z;
    c_X[1]=(_uv[1]-c_y)*f_y_1*z;
    c_X[2]=z;
    return c_X;
}

Eigen::Vector4f Camera::UnprojectInc(Eigen::Vector2f _uv, float _inv_depth)
{
    Eigen::Vector4f c_X;
    float z = 1.f/_inv_depth;
    c_X[0]=(_uv[0]-c_x)*f_x_1*z;
    c_X[1]=(_uv[1]-c_y)*f_y_1*z;
    c_X[2]=z;
    c_X[3]=1.f;
    return c_X;

}

Eigen::Vector2f Camera::ProjectDist(Eigen::Vector3f _X) {
    Eigen::Vector2f uv;
    float x_p = _X[0]/_X[2];
    float y_p = _X[1]/_X[2];
    float x_p_y_p = x_p*y_p;
    float x_p2 = x_p*x_p;
    float y_p2 = y_p*y_p;
    float r2=x_p2+y_p2;
    float r4=r2*r2;
    float dist_rad=1+dist[0]*r2+dist[1]*r4;
    uv[0] = f_x*(x_p*dist_rad+2*dist[2]*x_p_y_p+dist[3]*(r2+2*x_p2))+c_x;
    uv[1] = f_y*(y_p*dist_rad+2*dist[3]*x_p_y_p+dist[2]*(r2+2*y_p2))+c_y;

    return uv;
}

Eigen::Vector3f Camera::UnprojectDist(Eigen::Vector2f _uv, float _inv_depth) {
    Eigen::Vector3f X;

    return X;
}

Eigen::Vector4f Camera::UnprojectDistInc(Eigen::Vector2f _uv, float _inv_depth) {
    Eigen::Vector4f X;

    return X;
}

cv::Mat Camera::GetKcv()
{
    return cv_K.clone();
}

Eigen::Matrix3f Camera::GetK()
{
    return K;
}

cv::Mat Camera::GetDist()
{
    return cv_dist.clone();
}
