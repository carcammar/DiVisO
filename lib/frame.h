#ifndef FRAME_H
#define FRAME_H

#include <string>
#include <stdio.h>
#include <utility>
#include <iostream>
#include <mutex>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "Eigen/Dense"

#include "maths.h"
#include "camera.h"

class Point;

class Frame
{
public:
    Frame();
    Frame(unsigned int _id, const std::string _path_to_image, const unsigned int _scales, const double _scale_fact, const int _dist_negih, Camera* _p_cam);
    Frame(Frame* _p_fr);
    ~Frame();

    void Align(const Frame* _prev_fr);

    void GetPyrLevel(const unsigned int _lev, cv::Mat &_im);
    void GetGradLevel(const unsigned int _lev, bool _b_X, cv::Mat &_im);
    float GetIntPoint(const Eigen::Vector2f &_uv, const unsigned int _lev);
    void GetGradPoint(const Eigen::Vector2f &_uv, const unsigned int _lev, Eigen::Vector2f &_grad);

    void GetScaleFact(const unsigned _lev, Eigen::Vector2f &_scale_fact);
    void GetScaleFact(std::vector<Eigen::Vector2f> &_v_scale_fact);
    void GetScaleSize(const unsigned _lev, Eigen::Vector2i &_scale_size);
    void GetScaleSize(std::vector<Eigen::Vector2i> &_v_scale_size);

    void ChangeScalePoint(const unsigned int _scale_0, const unsigned int _scale_1, Eigen::Vector2f &_pt);

    void ExtractPoints(const unsigned int _n_points, const unsigned int _grid_rows, const unsigned int _grid_cols, const float _min_grad2);
    void AddPoint(Point* _p_pt);
    // void AddExtractedPoints(std::list<Point*> &_l_p_all_pts);

    void GetPointProjections(std::vector<cv::Point2f> &_v_pt_proj, std::vector<float> &_v_inv_depths);

    void SetPose(const Eigen::Matrix<float,4,4> _T_wc);
    Eigen::Matrix<float,4,4> GetPose();
    Eigen::Matrix<float,4,4> GetInvPose();


public:
    const unsigned int id;

    cv::Mat im;
    cv::Mat im_8u;
    cv::Mat im_und;
    cv::Mat feasible_pts;

    const unsigned int scales;
    cv::Mat mod_grad; // Just for finest level

    std::vector< std::pair<Eigen::Vector2f, std::vector<float> > > v_extr_points;

    bool b_KF;
    Camera* p_cam;




private: // should be private
    const double scale_fact;
    const int dist_neigh;

    std::mutex m_im; // For image TODO: not necessary since image is never modified
    std::vector<cv::Mat> v_pyramids;

    std::vector<Eigen::Vector2f> v_scale_fact; // real applied scaled factor when resizeing
    std::vector<Eigen::Vector2f> v_inv_scale_fact; // real applied scaled factor when resizeing

    std::vector<cv::Mat> v_gradX;
    std::vector<cv::Mat> v_gradY;

    std::mutex m_points; // mutex for Points seen from this frame
    std::list<Point*> l_points;

    std::mutex m_pose; // mutex for Points seen from this frame
    Eigen::Matrix<float,4,4> T_wc;
    Eigen::Matrix<float,4,4> T_cw;
};

#endif
