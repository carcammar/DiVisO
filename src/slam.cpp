
















//buscar:
// TODO FINISH
// TODO FINISH
// TODO FINISH
// TODO FINISH
// TODO FINISH
// TODO FINISH
// TODO FINISH
// TODO FINISH























#include "slam.h"


SLAM::SLAM(): curr_state(NOT_INITIALIZED), period(0.050f), curr_im_idx(0), p_curr_fr(nullptr),
    p_prev_fr(nullptr), scales(1), scale_fact(0.5), next_fr_id(0)
{
}

SLAM::SLAM(std::string _path_to_data, std::string _path_to_calibration):
    path_to_data_folder(_path_to_data), curr_state(NOT_INITIALIZED), curr_im_idx(0), p_curr_fr(nullptr),
    p_prev_fr(nullptr), next_fr_id(0)
{
    cv::FileStorage fs_calibration(_path_to_calibration.c_str(), cv::FileStorage::READ);
    if(!fs_calibration.isOpened())
    {
       std::cerr << "Failed to open calibation file at: " << _path_to_calibration << std::endl;
       exit(-1);
    }

    f[0] = fs_calibration["Camera.fx"];
    f[1] = fs_calibration["Camera.fy"];
    c[0] = fs_calibration["Camera.cx"];
    c[1] = fs_calibration["Camera.cy"];

    dist[0] = fs_calibration["Camera.k1"];
    dist[1] = fs_calibration["Camera.k2"];
    dist[2] = fs_calibration["Camera.p1"];
    dist[3] = fs_calibration["Camera.p2"];

    period = fs_calibration["Camera.fps"];
    period = 1.f/period;

    scales = round(fs_calibration["Frame.scales"]);
    scale_fact = fs_calibration["Frame.scale_factor"];
    grid_rows = round(fs_calibration["Frame.rows"]);
    grid_cols = round(fs_calibration["Frame.cols"]);
    n_points = round(fs_calibration["Frame.points"]);
    min_grad = fs_calibration["Frame.min_gradient"];
    dist_neigh = round(fs_calibration["Frame.dist_neigh"]);
    min_grad_2 = min_grad*min_grad;

    std::cout << "Camera matrix: " << f[0] <<", " << f[1] <<", " << c[0] <<", " << c[1] << std::endl;
    std::cout << "Distortion coefficients: " << dist[0] <<", " << dist[1] <<", " << dist[0] <<", " << dist[1] << std::endl;
    std::cout << "Grid: " << grid_rows << "x" << grid_cols << std::endl;
    std::cout << "Minimun gradient: " << min_grad << std::endl;
    std::cout << "Points: " << n_points << std::endl;

    // Create camera attribute
    p_cam = new Camera(f, c, dist);

    // Create displayer thread
    p_displayer = new Display(this);
    pt_displayer = new std::thread(&Display::Run, p_displayer);

    // Points
    l_all_points.clear();
    // l_all_points.reserve(20000); // TODO check this size

    // Keyframes
    l_all_KFs.clear();
    // l_all_KFs.resize(1000); // TODO check this size
}

/*void SLAM::LoadImage(){

    curr_im = cv::imread(path_to_data_folder+"/"+std::to_string(curr_im_idx)+".png");
    // buildPyramid();
    b_update=true;
}*/

void SLAM::Initialize()
{
    p_curr_fr->ExtractPoints(n_points, grid_rows, grid_cols, min_grad_2);
    p_curr_fr->SetPose(Eigen::MatrixXf::Identity(4,4));

    // Initialize extracted points with random depth
    std::unique_lock<std::mutex> lock(m_points);

    const float min_inv_depth = 0.05f;
    const float max_inv_depth = 0.5f;
    const float dif_inv_depth = max_inv_depth - min_inv_depth;
    float inv_depth;
    for(auto it=p_curr_fr->v_extr_points.begin(); it!=p_curr_fr->v_extr_points.end(); it++)
    {
        // random inverse depth
        inv_depth = min_inv_depth + dif_inv_depth*( static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        float _uv[2];
        _uv[0]=(*it).first[0];
        _uv[1]=(*it).first[1];
        Point* p_point = new Point(p_curr_fr, _uv, inv_depth, (*it).second, std::vector<Frame*>());
        // Add point to frame current frame
        p_curr_fr->AddPoint(p_point);
        // Add to SLAM points list
        l_all_points.push_back(p_point);
    }

    ConvertToKF(p_curr_fr);
    curr_state=NOMINAL;
}


void SLAM::Run(){
    while(1){

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Load and process image
        p_curr_fr = new Frame(++next_fr_id, path_to_data_folder+"/"+std::to_string(curr_im_idx)+".png",
                              scales, scale_fact, dist_neigh, p_cam);

        std::cout << "p_curr_fr->id: " << p_curr_fr->id << std::endl;
        // Do tracking
        if (curr_state==NOT_INITIALIZED){
            std::cout << "Initializing " << std::endl;
            Initialize();
        }
        else if(curr_state==NOMINAL){
            std::cout << "Points in map: " << l_all_points.size() << std::endl;
            std::cout << "KFs in map: " << l_all_KFs.size() << std::endl;

            // cv::waitKey();
            Tracking();
        }
        else if(curr_state==LOST)
        {
            // Try to recover
            break;
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        float tracking_time= std::chrono::duration_cast<std::chrono::duration<float> >(t2 - t1).count();
        std::cout << "Tracking time: " << tracking_time << std::endl;

        if(tracking_time<period)
            usleep(static_cast<__useconds_t>((period-tracking_time)*1e6f));

        // Update drawer
        cv::Mat im_to_display;
        // p_curr_fr->GetGradLevel(3, true, im_to_display);
        // im_to_display = p_curr_fr->mod_grad.clone();
        p_curr_fr->GetPyrLevel(0, im_to_display);
        im_to_display.convertTo(im_to_display,CV_8UC1);

        {
            std::unique_lock<std::mutex> lock(m_frames);
            std::unique_lock<std::mutex> lock2(m_points);

            // TODO FINISH: Solve conflict mutex between Displayer->Update/frame->GetPose and Displayer->Run()
            // Check the use of mutex in the whole program
            p_displayer->Update(im_to_display, p_curr_fr, l_all_points, l_all_KFs);
        }


        // Update current frame
        delete p_prev_fr;

        p_prev_fr = p_curr_fr;
        p_curr_fr = nullptr;

        // cv::waitKey();
        usleep(static_cast<__useconds_t>(1e6f));

        curr_im_idx++;
    }
}

void SLAM::Tracking()
{
    // TODO FINISH

    p_curr_fr->SetPose(p_prev_fr->GetPose());

    // Start with image alignment
}

void SLAM::ConvertToKF(Frame* _kf)
{
    _kf->b_KF=true;
    std::unique_lock<std::mutex> lock(m_frames);
    l_all_KFs.push_back(_kf);
}
