#ifndef DISPLAY_H
#define DISPLAY_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <pangolin/pangolin.h>

#include <mutex>
#include <chrono>
#include <unistd.h>

#include "frame.h"
#include "point.h"

class SLAM;

class Display
{
public:
    Display(SLAM* _p_SLAM);
    void Run();
    /*void Update(const cv::Mat &_im, const std::vector<cv::Point2f> &_v_pts=std::vector<cv::Point2f>(),
                const std::vector<float> &_v_pts_inv_depth=std::vector<float>()); // Update to show other image than frame with points
    */
    void Update(const cv::Mat &_im, Frame* _p_curr_fr, const std::list<Point*> &_l_points, const std::list<Frame*> &_l_KFs);

    // void UpdateFrame(); // Update to show frame and points
    void Stop();

private:
    SLAM* p_SLAM;

    bool b_stop;
    bool b_update;

    pangolin::OpenGlMatrix Twc;
    void DrawCamera(Eigen::Matrix4f _Twc);

    const float w_cam = 0.2f;
    const float z_cam = 0.2f;
    const float h_cam = 0.2f;

    double mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    std::mutex mut_im;
    cv::Mat im;
    std::vector<cv::Point2f> v_pts;
    std::vector<float> v_pts_inv_depth;

    std::mutex mut_map;
    std::vector<Eigen::Vector3f> v_3D_pts;
    std::vector<Eigen::Matrix4f> v_Twc_KFs;
    Eigen::Matrix4f curr_Twc;
};

#endif
