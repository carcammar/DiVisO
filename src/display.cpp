#include "display.h"

Display::Display(SLAM* _p_SLAM): b_stop(false), b_update(true), p_SLAM(_p_SLAM)
{
    mViewpointX = 0;
    mViewpointY = -0.7;
    mViewpointZ = -1.8;
    mViewpointF = 500;

}

void Display::Run(){
    // For current frame
    cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);

    // For map (TODO check all these values have been directly copied from ORBSLAM)
    pangolin::CreateWindowAndBind("Map",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0/768.0)
            .SetHandler(new pangolin::Handler3D(s_cam));

    Twc.SetIdentity();

    while(!b_stop){
        // Draw 3D scene
        // glClearColor(1.0f,1.0f,1.0f,1.0f); // Draw white background

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        s_cam.Follow(Twc);
        d_cam.Activate(s_cam);
        glPointSize(5);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);
        Eigen::Vector3f aux;

        std::cout << "map drawer" << std::endl;
        // TODO FINISH avoid a so big mutex, too much time
        //while(!mut_map.try_lock())
        {
            std::unique_lock<std::mutex> lock(mut_map);
            //mut_map.lock();
            std::cout << "lock map" << std::endl;

            for(std::vector<Eigen::Vector3f>::const_iterator itPt=v_3D_pts.begin(); itPt!=v_3D_pts.end(); itPt++)
            {
                glVertex3f((*itPt)[0],(*itPt)[1],(*itPt)[2]);
            }
            glEnd();
            //mut_map.unlock();
            std::cout << "unlock map" << std::endl;

            // glBegin(GL_LINES);
            Eigen::Matrix4f _Twc;
            cv::Mat cvTwc;
            // Draw keyframes (TODO with method drawcamera)
            for(auto itKF=v_Twc_KFs.begin(); itKF!=v_Twc_KFs.end(); itKF++)
            {
                _Twc=(*itKF);
                cvTwc = Maths::Eigmat2Cvmat(_Twc).t();
                std::cout << "cvTwc: " << cvTwc << std::endl;
                // TODO why transpose? Ti works...
                glPushMatrix();
                glMultMatrixf(cvTwc.ptr<GLfloat>(0));
                glLineWidth(5);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w_cam,h_cam,z_cam);
                glVertex3f(0,0,0);
                glVertex3f(w_cam,-h_cam,z_cam);
                glVertex3f(0,0,0);
                glVertex3f(-w_cam,-h_cam,z_cam);
                glVertex3f(0,0,0);
                glVertex3f(-w_cam,h_cam,z_cam);

                glVertex3f(w_cam,h_cam,z_cam);
                glVertex3f(w_cam,-h_cam,z_cam);

                glVertex3f(-w_cam,h_cam,z_cam);
                glVertex3f(-w_cam,-h_cam,z_cam);

                glVertex3f(-w_cam,h_cam,z_cam);
                glVertex3f(w_cam,h_cam,z_cam);

                glVertex3f(-w_cam,-h_cam,z_cam);
                glVertex3f(w_cam,-h_cam,z_cam);
                glEnd();

                glPopMatrix();
            }

            _Twc=curr_Twc;
            cvTwc = Maths::Eigmat2Cvmat(_Twc).t();
            std::cout << "_Twc: " << _Twc << std::endl;
            std::cout << "cvTwc: " << cvTwc << std::endl;
            // TODO why transpose? Ti works...
            glPushMatrix();
            glMultMatrixf(cvTwc.ptr<GLfloat>(0));
            glLineWidth(5);
            glColor3f(0.0f,1.0f,0.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w_cam,h_cam,z_cam);
            glVertex3f(0,0,0);
            glVertex3f(w_cam,-h_cam,z_cam);
            glVertex3f(0,0,0);
            glVertex3f(-w_cam,-h_cam,z_cam);
            glVertex3f(0,0,0);
            glVertex3f(-w_cam,h_cam,z_cam);

            glVertex3f(w_cam,h_cam,z_cam);
            glVertex3f(w_cam,-h_cam,z_cam);

            glVertex3f(-w_cam,h_cam,z_cam);
            glVertex3f(-w_cam,-h_cam,z_cam);

            glVertex3f(-w_cam,h_cam,z_cam);
            glVertex3f(w_cam,h_cam,z_cam);

            glVertex3f(-w_cam,-h_cam,z_cam);
            glVertex3f(w_cam,-h_cam,z_cam);
            glEnd();

            glPopMatrix();

            // Draw current frame
            glEnd();
        }

        pangolin::FinishFrame();

        // Draw current frame and tracks
        {
            std::unique_lock<std::mutex> lock(mut_im);

            if (im.empty())
                continue;

            const float max_inv_depth = 10.f;
            const float min_inv_depth = 0.1f;
            const float dif_inv_depth = max_inv_depth-min_inv_depth;

            // TODO FINISH display projected points with colour standing for its inverse depth
            size_t i=0;
            for(std::vector<cv::Point2f>::iterator itPt=v_pts.begin(); itPt!=v_pts.end(); itPt++, i++)
            {
                // std::cout << "inv depth: " << v_pts_inv_depth[i] << std::endl;
                float norm_depth = 255*(v_pts_inv_depth[i]-min_inv_depth)/dif_inv_depth;
                // std::cout << "norm_depth: " << norm_depth << std::endl;
                norm_depth=std::max(norm_depth, 0.f);
                norm_depth=std::min(norm_depth, 255.f);

                cv::circle(im, (*itPt), 3, CV_RGB(0, round(norm_depth), round(norm_depth)), 1, cv::LINE_8);
            }

            cv::imshow("Test", im);
        }

        cv::waitKey(30);
    }
}

void Display::Update(const cv::Mat &_im, Frame* _p_curr_fr, const std::list<Point*> &_l_points, const std::list<Frame*> &_l_KFs)
{
    std::cout << "Update" << std::endl;
    // Update image
    // while (!mut_im.try_lock())
    {
        std::cout << "Update image" << std::endl;
        std::cout << "mut_im locked (update)" << std::endl;

        std::unique_lock<std::mutex> lock(mut_im);
        _im.copyTo(im);
        cv::cvtColor(im, im, CV_GRAY2RGB);
        // Update projected points
        _p_curr_fr->GetPointProjections(v_pts, v_pts_inv_depth);
        // mut_im.unlock();
        std::cout << "mut_im unlocked (update)" << std::endl;

    }

    //while (!mut_map.try_lock())
    {
        std::cout << "Update map" << std::endl;
        std::cout << "mut_map locked (update)" << std::endl;

        std::unique_lock<std::mutex> lock(mut_map);
        // Update 3D points
        v_3D_pts.resize(_l_points.size());
        size_t i=0;
        std::cout << "Points for diplayer: " << _l_points.size() << std::endl;
        for(std::list<Point*>::const_iterator itPt=_l_points.begin(); itPt!=_l_points.end(); itPt++, i++)
        {
            if (!*itPt)
                continue;
            v_3D_pts[i]=(*itPt)->GetWorldPos();
            std::cout << "Point position: " << v_3D_pts[i] << ": " << std::endl;

        }

        // Update keyframes
        v_Twc_KFs.resize(_l_KFs.size());
        std::cout << "KFs for displayer: " << _l_KFs.size() << std::endl;
        i=0;
        for(std::list<Frame*>::const_iterator itKF=_l_KFs.begin(); itKF!=_l_KFs.end(); itKF++)
        {
            if (!*itKF)
                continue;
            v_Twc_KFs[i]=(*itKF)->GetPose();
            std::cout << "Pose KF " << i << ": " << v_Twc_KFs[i] << std::endl;
        }

        // update current pose
        curr_Twc = _p_curr_fr->GetPose();
        //std::cout << "mut_map unlocked (update)" << std::endl;
        //mut_map.unlock();
    }
}

void Display::Stop(){
    b_stop=true;
}

void Display::DrawCamera(Eigen::Matrix4f _Twc){
    cv::Mat cvTwc = Maths::Eigmat2Cvmat(_Twc).t();
    // TODO why transpose? Ti works...
    glPushMatrix();
    glMultMatrixf(cvTwc.ptr<GLfloat>(0));
    glLineWidth(5);
    glColor3f(0.0f,0.0f,1.0f);
    glVertex3f(0,0,0);
    glVertex3f(w_cam,h_cam,z_cam);
    glVertex3f(0,0,0);
    glVertex3f(w_cam,-h_cam,z_cam);
    glVertex3f(0,0,0);
    glVertex3f(-w_cam,-h_cam,z_cam);
    glVertex3f(0,0,0);
    glVertex3f(-w_cam,h_cam,z_cam);

    glVertex3f(w_cam,h_cam,z_cam);
    glVertex3f(w_cam,-h_cam,z_cam);

    glVertex3f(-w_cam,h_cam,z_cam);
    glVertex3f(-w_cam,-h_cam,z_cam);

    glVertex3f(-w_cam,h_cam,z_cam);
    glVertex3f(w_cam,h_cam,z_cam);

    glVertex3f(-w_cam,-h_cam,z_cam);
    glVertex3f(w_cam,-h_cam,z_cam);
    glEnd();

    glPopMatrix();
}
