#ifndef SLAM_H
#define SLAM_H

#include <thread>
#include <string>
#include <chrono>
#include <unistd.h>

#include <Eigen/Dense>

#include "display.h"
#include "frame.h"
#include "point.h"
#include "camera.h"

class SLAM;
/*class Display
{
public:
    Display(SLAM* _p_SLAM);
    void Run();
};*/

class SLAM
{
public:
    SLAM();
    SLAM(std::string _path_to_data, std::string _path_to_calibration);
    // void LoadImage();
    void Initialize();
    void Run();

public:
    // Pointers to last two frames
    Frame* p_curr_fr;
    Frame* p_prev_fr;

private:
    void Tracking();
    void ConvertToKF(Frame* _kf);

private:
    enum state{
        NOT_INITIALIZED,
        NOMINAL,
        LOST
    };

    const std::string path_to_data_folder;
    Display* p_displayer;
    std::thread* pt_displayer;

    bool b_update;
    state curr_state;

    float period;
    cv::Mat curr_im;
    unsigned int curr_im_idx;

    unsigned int next_fr_id;

    // camera parameters
    float f[2]; // f_x, f_y
    float c[2]; // c_x, c_y
    float dist[4]; // k_1, k_2, p_1, p_2
    Camera* p_cam;

    // Frame parameters
    unsigned int scales;
    double scale_fact;

    // Point extraction parameters
    unsigned int grid_rows;
    unsigned int grid_cols;
    unsigned int n_points;
    float min_grad, min_grad_2;
    int dist_neigh;

    // Map elements
    // Points
    std::list<Point*> l_all_points;
    // Keyframes
    std::list<Frame*> l_all_KFs;
    // mutex to be called when map is modified
    std::mutex m_points;
    std::mutex m_frames;
    std::mutex m_map;
};



#endif // SLAM_H
