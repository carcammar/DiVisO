
















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
    p_prev_fr(nullptr), scales(1), scale_fact(0.5), next_fr_id(0), p_init_fr(nullptr)
{
}

SLAM::SLAM(std::string _path_to_data, std::string _path_to_calibration):
    path_to_data_folder(_path_to_data), curr_state(NOT_INITIALIZED), curr_im_idx(0), p_curr_fr(nullptr),
    p_prev_fr(nullptr), next_fr_id(0), p_init_fr(nullptr)
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

    orb_detector = cv::ORB::create(500,1.2f,1); // Modify arguments
    orb_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
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
            Initialize2Fr();
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


void SLAM::Initialize1Fr()
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

void SLAM::Initialize2Fr()
{
    const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
    const double ransac_thresh = 2.5f; // RANSAC inlier threshold

    // 1. Select initial frame
    if (!p_init_fr)
    {
        p_init_fr = new Frame(p_curr_fr);
        // TODO use grid for point extraction
        orb_detector->detectAndCompute(p_init_fr->im_8u, cv::noArray(), v_kp_init, init_desc);
        return;
    }

    // 2. Find motion between initial frame and current frame
    // 2.1 Points for current frame
    Eigen::Matrix<float,4,4> T_01;
    // TODO use grid for point extraction
    orb_detector->detectAndCompute(p_curr_fr->im_8u, cv::noArray(), v_kp_curr, curr_desc);
    std::cout << "init_desc.size() " << init_desc.size() << std::endl;
    std::cout << "curr_desc.size() " << curr_desc.size() << std::endl;
    init_matches.clear();
    orb_matcher->knnMatch(init_desc, curr_desc, init_matches, 2);

    // 2.2 Match two frames
    v_matched_p_init.clear();
    v_matched_p_curr.clear();
    std::vector<cv::DMatch> good_matches;
    v_matched_p_init.reserve(v_kp_init.size());
    v_matched_p_curr.reserve(v_kp_curr.size());
    good_matches.reserve(v_kp_curr.size());

    for(unsigned i = 0; i < init_matches.size(); i++)
    {
        if(init_matches[i][0].distance < nn_match_ratio*init_matches[i][1].distance)
        {
            good_matches.push_back(init_matches[i][0]);
            v_matched_p_init.push_back(v_kp_init[init_matches[i][0].queryIdx].pt);
            v_matched_p_curr.push_back(v_kp_curr[init_matches[i][0].trainIdx].pt);
            // std::cout << v_kp_init[init_matches[i][0].queryIdx].pt << "-" << v_kp_curr[init_matches[i][0].trainIdx].pt << std::endl;
        }
    }

    // 2.3 Find essential matrix between two frames
    cv::Mat inlier_mask;
    cv::Mat K =  p_curr_fr->p_cam->GetKcv();
    cv::Mat fund_mat;
    cv::findFundamentalMat(v_matched_p_init, v_matched_p_curr, CV_FM_RANSAC , 3, 0.99, inlier_mask).convertTo(fund_mat, CV_32F);
    // std::cout << "fund_mat = " << fund_mat << std::endl;
    cv::Mat ess_mat = K.t()*fund_mat*K;

    cv::Mat A = cv::Mat::zeros(3,3,CV_32F);
    A.at<float>(0,1) = -1.f;
    A.at<float>(1,0) = 1.f;
    A.at<float>(2,2) = 1.f;
    cv::Mat B = cv::Mat::zeros(3,3,CV_32F);
    B.at<float>(0,1) = 1.f;
    B.at<float>(1,0) = -1.f;

    cv::Mat t_init = cv::Mat::zeros(3,1,CV_32F);
    cv::Mat R_init1, R_init2;
    cv::Mat W, U, Vt;
    cv::SVD cv_svd;
    cv_svd.compute(ess_mat, W, U, Vt);
    cv::Mat Tx = U*B*U.t();
    t_init.at<float>(0) = Tx.at<float>(2,1);
    t_init.at<float>(1) = Tx.at<float>(0,2);
    t_init.at<float>(2) = Tx.at<float>(1,0);
    R_init1 = U*A*Vt;
    R_init2 = U*A.t()*Vt;

    /*std::cout << "Tx = " << Tx << std::endl; // t_01
    std::cout << "t_init = " << t_init << std::endl; // t_01
    std::cout << "R_init1 = " << R_init1 << std::endl; // R_01
    std::cout << "R_init2 = " << R_init2 << std::endl; // R_01*/

    /*cv::Mat img_matches;
    cv::drawMatches(p_init_fr->im_8u, v_kp_init, p_curr_fr->im_8u, v_kp_curr, good_matches, img_matches, cv::Scalar::all(-1),
                         cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("Good Matches", img_matches );
    cv::waitKey();*/


    // 2.4 Compute initial solution with positive depth
    cv::Mat P1 = cv::Mat::zeros(3,4,CV_32F);
    P1.rowRange(0,3).colRange(0,3) = cv::Mat::eye(3,3,CV_32F);
    P1 = K*P1;
    cv::Mat P2(3,4,CV_32F);
    cv::Mat R_test(3,3,CV_32F);
    cv::Mat t_test(3,3,CV_32F);
    int good_pts[4];
    int max_good=0;
    cv::Mat R_good;
    cv::Mat t_good;
    int idx_good;
    for(int count=0; count<4; count++)
    {
        good_pts[count] = 0;
        if (count == 0)
        {
            R_test = R_init1.clone();
            t_test = t_init.clone();
        }
        else if (count == 1)
        {
            R_test = R_init1.clone();
            t_test = -t_init.clone();
        }
        else if (count == 2)
        {
            R_test = R_init2.clone();
            t_test = t_init.clone();
        }
        else
        {
            R_test = R_init2.clone();
            t_test = -t_init.clone();
        }


        P2.rowRange(0,3).colRange(0,3) = R_test.clone();
        P2.rowRange(0,3).colRange(3,4) = t_test.clone();

        R_test.copyTo(P2.rowRange(0,3).colRange(0,3));
        t_test.copyTo(P2.rowRange(0,3).col(3));
        P2 = K*P2;

        // cv::Mat t_10 = -R_test.t()*t_test;
        cv::Mat p3d1;

        /*std::cout << "P1 = " << P1 << std::endl;
        std::cout << "P2 = " << P2 << std::endl;*/

        // unsigned int i = 0;
        std::vector<cv::Point2f>::iterator itPt2 = v_matched_p_curr.begin();

        for(unsigned int i=0; i < v_matched_p_init.size(); i++)
        {
            if(inlier_mask.at<uchar>(i))
            {
                // Triangulate((*itPt1), (*itPt2), P1, P2, p3d1);
                Triangulate(v_matched_p_init[i], v_matched_p_curr[i], P1, P2, p3d1);

                // Check positive depth in both frames and parallax
                cv::Mat p3d2 = R_test*p3d1+t_test;
                /*std::cout << "pt1 = " << v_matched_p_init[i] << std::endl << "pt2 = " << v_matched_p_curr[i] << std::endl;
                std::cout << "p3d1 = " << p3d1.t() << std::endl << "p3d2 = " << p3d2.t() << std::endl;*/
                if ((p3d1.at<float>(2) > 0.f) && (p3d2.at<float>(2) > 0.f))
                    good_pts[count]++;
            }
        }

        // TODO
        if(good_pts[count]>=max_good)
        {
            idx_good=count;
            max_good=good_pts[count];
            R_good = R_test.clone();
            t_good = t_test.clone();
        }

        std::cout << "good " << count << ": " << good_pts[count] << "/" << v_matched_p_init.size() << std::endl;
    }


    // TODO 2.5 Check parallax
    float parallax = 1.f; // dist/median_depth
    // Compute parallax
    if (parallax<0.05f)
        return;

    // 3. Triangulate points of high gradient, and initialize them
    Eigen::Matrix<float,4,4> T10 = Eigen::Matrix4f::Identity();
    T10.block<3,3>(0,0)=Maths::Cvmat2Eigmat(R_good);
    T10.block<3,1>(0,3)=Maths::Cvmat2Eigmat(t_good);
    EpipolarTriangulate(p_init_fr, p_curr_fr, T10);
}



void SLAM::Tracking()
{
    // TODO FINISH

    p_curr_fr->SetPose(p_prev_fr->GetPose());

    // Start with image alignment
}

void SLAM::ImageAlignment()
{
    /*
     * Align current frame w.r.t previous frame and its points with estimated depth
     */

    // TODO FINISH

}


void SLAM::ConvertToKF(Frame* _kf)
{
    _kf->b_KF=true;
    std::unique_lock<std::mutex> lock(m_frames);
    l_all_KFs.push_back(_kf);
}

void SLAM::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = p1.x*P1.row(2)-P1.row(0);
    A.row(1) = p1.y*P1.row(2)-P1.row(1);
    A.row(2) = p2.x*P2.row(2)-P2.row(0);
    A.row(3) = p2.y*P2.row(2)-P2.row(1);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = Vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

// TODO inline this function
void SLAM::ComputeEpipolarLine(const Eigen::Vector2f &_uv, const Eigen::Matrix<float,4,4> &_T10, const Eigen::Matrix<float,3,3> &_K, const float _depth_0, Eigen::Vector2f &_dir_epi, Eigen::Vector2f &_o_epi)
{
    Eigen::Vector3f uv1;
    Eigen::Matrix<float,3,3> K_1 = _K.inverse(); // TODO check this inverse

    uv1 << _uv(0), _uv(1), 1.f;

    Eigen::Vector3f X0 = K_1*uv1;
    Eigen::Vector3f X0_p = X0;
    X0 *= _depth_0/X0(2);
    X0_p *= 1.01f*_depth_0/X0_p(2);
    Eigen::Vector3f X1 = _K*(_T10.block<3,3>(0,0)*X0+_T10.block<3,1>(0,3));
    Eigen::Vector3f X1_p = _K*(_T10.block<3,3>(0,0)*X0_p+_T10.block<3,1>(0,3));
    X1 /= X1(2);
    X1_p /= X1_p(2);
    _dir_epi = X1_p.block<2,1>(0,0)-X1.block<2,1>(0,0);
    _dir_epi /= _dir_epi.norm();
    _o_epi = X1.block<2,1>(0,0);
}

void SLAM::EpipolarTriangulate(Frame* _p_fr_0, Frame* _p_fr_1, const Eigen::Matrix<float,4,4> &_T10)
{
    Eigen::Matrix3f K_cam = _p_fr_0->p_cam->GetK();
    std::cout << "_T10 = " << _T10 << std::endl;
    std::cout << "K_cam = " << K_cam << std::endl;

    float depth_0 = 100.f;
    Eigen::Vector2f dir_epi;
    Eigen::Vector2f o_epi;
    Eigen::Vector2f grad;
    std::vector<Eigen::Vector2f> v_scale_fact;
    _p_fr_0->GetScaleFact(v_scale_fact);

    // 0. Extract points of high gradient in first frame
    _p_fr_0->ExtractPoints(n_points, grid_rows, grid_cols, min_grad_2);

    // 1. Triangulate each extracted point
    const unsigned int scales = _p_fr_0->scales;
    float I_0, I_1, g_1;
    Eigen::Vector2f uv_0, uv_1;

    for (auto it = _p_fr_0->v_extr_points.begin(); it != _p_fr_0->v_extr_points.end(); it++)
    {
        // 1.1 Compute epipolar line
        uv_0 = it->first;
        ComputeEpipolarLine(uv_0, _T10, K_cam, depth_0, dir_epi, uv_1);
        _p_fr_0->ChangeScalePoint(0,scales, uv_0);
        _p_fr_1->ChangeScalePoint(0,scales, uv_1);

        std::cout << "uv_0 = " << uv_0.transpose() << std::endl << "    uv_1 = " << uv_1.transpose() << std::endl;


        // 1.2 Search along epipolar line
        for(int lev=scales; lev>=0; lev--)
        {
            // std::cout << "   scale: " << lev << "    scale_fact: " << v_scale_fact[lev].transpose() << "      point: " << uv_1.transpose() << "     grad: " << g_1 << std::endl;
            I_0 = _p_fr_0->GetIntPoint(uv_0, lev);
            I_1 = _p_fr_1->GetIntPoint(uv_1, lev);

            _p_fr_1->GetGradPoint(uv_1, lev, grad);
            g_1 = grad.dot(dir_epi);

            if(lev==scales)
                std::cout << "   scale: " << lev << "    I_0-I_1: " << I_0-I_1 << "      point: " << uv_1.transpose() << "     grad: " << g_1 << std::endl;

            uv_1 += dir_epi*(I_0-I_1)/g_1; // update
            std::cout << "   scale: " << lev << "    I_0-I_1: " << I_0-I_1 << "      point: " << uv_1.transpose() << "     grad: " << g_1 << std::endl;

            if (lev>0)
            {
                _p_fr_0->ChangeScalePoint(lev,lev-1, uv_0);
                _p_fr_1->ChangeScalePoint(lev,lev-1, uv_1);
            }

        }


    }
}
