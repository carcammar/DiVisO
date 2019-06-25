#ifndef POINT_H
#define POINT_H

#include <vector>


#include "Eigen/Dense"

#include "frame.h"

class Point
{
public:
    Point();
    Point(Frame* _p_host_fr, float *_uv_host_fr, float _inv_depth,
          std::vector<float> _v_neigh_pixels, std::vector<Frame*> _v_obs_frs);

    void Project(const Eigen::Matrix<float,4,4> _T_wc, Eigen::Vector2f &_uv_c);
    // void Project(Frame* _p_fr, Eigen::Vector2f &_uv_c);
    void Project(Frame* _p_fr, Eigen::Vector2f &_uv_c, float &_inv_depth);
    void Project(Frame* _p_fr, Eigen::Vector2f &_uv_c);

    unsigned int GetHostId();
    Eigen::Vector3f GetWorldPos();
public:

private:


private:
    // Paremeters wrt host frame
    Frame* p_host_fr;
    Eigen::Vector2f uv_host_fr;
    float inv_depth;
    Eigen::Vector3f w_xyz; // xyz w.r.t world (TODO: Update after enter in local map)

    std::vector<float> v_neigh_pixels;
    std::vector<Frame*> v_obs_frs;

};

#endif
