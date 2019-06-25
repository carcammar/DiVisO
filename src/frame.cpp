#include "frame.h"
#include "point.h"

Frame::Frame(unsigned int _id, const std::string _path_to_image, const unsigned int _scales, const double _scale_fact, const int _dist_negih, Camera* _p_cam):
    id(_id), scales(_scales), scale_fact(_scale_fact), dist_neigh(_dist_negih), b_KF(false), p_cam(_p_cam)
{
    // TODO: Parallelize this method
    std::unique_lock<std::mutex> lock(m_im);

    cv::imread(_path_to_image, CV_LOAD_IMAGE_GRAYSCALE).convertTo(im, CV_32F);
    if (im.empty())
        std::cout << "Error: EMPTY IMAGE" << std::endl;

    // TODO: ideally remove undistortion
    std::cout << "Cam matrix: " << p_cam->GetK() << std::endl;
    std::cout << "Dist parameters: " << p_cam->GetDist() << std::endl;
    std::cout << "im size: " << im.size() << std::endl;
    cv::undistort(im.clone(), im, p_cam->GetK(), p_cam->GetDist());

    v_pyramids.resize(scales+1);
    v_gradX.resize(scales+1);
    v_gradY.resize(scales+1);

    v_pyramids[0]=im.clone(); // TODO: clone?

    // Compute pyramids and gradients
    for(unsigned int i=0; i<=scales; i++){
        const int rows = v_pyramids[i].rows;
        const int cols = v_pyramids[i].cols;

        std::cout << "(c,r) = " << cols << ", " << rows << std::endl;

        cv::Mat im_1_2 = 0.5*v_pyramids[i];
        cv::Mat im_1_4 = 0.5*im_1_2;
        cv::Mat im_1_8 = 0.5*im_1_4;
        cv::Mat im_1_16 = 0.5*im_1_8;

        // 1. Gradient (* for pose to be computed)
        // 1.1 Gradient at X [-1/2*,1/2]
        v_gradX[i] = cv::Mat(v_pyramids[i].size(), CV_32F);
        v_gradX[i](cv::Rect(0,0,cols-1,rows)) = im_1_2(cv::Rect(1,0,cols-1,rows)) - im_1_2(cv::Rect(0,0,cols-1,rows));
        // last column
        v_gradX[i].col(cols-1)=v_gradX[i].col(cols-2).clone();

        // 1.2 Gradient at Y [-1/2*,1/2]^T
        v_gradY[i] = cv::Mat(v_pyramids[i].size(), CV_32F);
        v_gradY[i](cv::Rect(0,0,cols,rows-1)) = im_1_2(cv::Rect(0,1,cols,rows-1)) - im_1_2(cv::Rect(0,0,cols,rows-1));
        // last row
        v_gradY[i].row(rows-1)=v_gradX[i].row(rows-2).clone();

        // std::cout << v_gradX[i] << std::endl;

        // 2. Blur [1/16, 1/8, 1/16; 1/8, 1/4*, 1/8; 1/16, 1/8, 1/16]
        cv::Mat im_blur = cv::Mat(v_pyramids[i].size(), CV_32F);
        im_blur(cv::Rect(1,1,cols-2,rows-2))=
                im_1_16(cv::Rect(0,0,cols-2,rows-2))+im_1_8(cv::Rect(1,0,cols-2,rows-2))+im_1_16(cv::Rect(2,0,cols-2,rows-2))+
                im_1_8(cv::Rect(0,1,cols-2,rows-2))+im_1_4(cv::Rect(1,1,cols-2,rows-2))+im_1_8(cv::Rect(2,1,cols-2,rows-2))+
                im_1_16(cv::Rect(0,2,cols-2,rows-2))+im_1_8(cv::Rect(1,2,cols-2,rows-2))+im_1_16(cv::Rect(2,2,cols-2,rows-2));

        // 2.1 First row [1/8,1/2*,1/8;1/16,1/8,1/16]
        im_blur(cv::Rect(1,0,cols-2,1))=
                im_1_8(cv::Rect(0,0,cols-2,1))+im_1_4(cv::Rect(1,0,cols-2,1))+im_1_8(cv::Rect(2,0,cols-2,1))+
                im_1_16(cv::Rect(0,1,cols-2,1))+im_1_8(cv::Rect(1,1,cols-2,1))+im_1_16(cv::Rect(2,1,cols-2,1));

        // 2.2 First column [1/8,1/16;1/2*,1/8;1/8,1/16]
        im_blur(cv::Rect(0,1,1,rows-2))=
                im_1_8(cv::Rect(0,0,1,rows-2))+im_1_16(cv::Rect(1,0,1,rows-2))+
                im_1_2(cv::Rect(0,1,1,rows-2))+im_1_8(cv::Rect(1,1,1,rows-2))+
                im_1_8(cv::Rect(0,2,1,rows-2))+im_1_16(cv::Rect(1,2,1,rows-2));

        // 2.3 Last column [1/16,1/8;1/8,1/2*;1/16,1/8]
        im_blur(cv::Rect(cols-1,1,1,rows-2))=
                im_1_16(cv::Rect(cols-2,0,1,rows-2))+im_1_8(cv::Rect(cols-1,0,1,rows-2))+
                im_1_8(cv::Rect(cols-2,1,1,rows-2))+im_1_2(cv::Rect(cols-1,1,1,rows-2))+
                im_1_8(cv::Rect(cols-2,2,1,rows-2))+im_1_16(cv::Rect(cols-1,2,1,rows-2));

        // 2.4 Last row [1/16,1/8,1/16;1/8,1/2*,1/8]
        im_blur(cv::Rect(1,rows-1,cols-2,1))=
                im_1_16(cv::Rect(0,rows-2,cols-2,1))+im_1_8(cv::Rect(1,rows-2,cols-2,1))+im_1_16(cv::Rect(2,rows-2,cols-2,1))+
                im_1_8(cv::Rect(0,rows-1,cols-2,1))+im_1_2(cv::Rect(1,rows-1,cols-2,1))+im_1_8(cv::Rect(2,rows-1,cols-2,1));

        // 2.5 Corners
        im_blur.at<float>(0,0)=im_1_2.at<float>(0,0)+im_1_4.at<float>(1,0)+im_1_4.at<float>(0,1);
        im_blur.at<float>(rows-1,0)=im_1_2.at<float>(rows-1,0)+im_1_4.at<float>(rows-2,0)+im_1_4.at<float>(rows-1,1);
        im_blur.at<float>(0,cols-1)=im_1_2.at<float>(0,cols-1)+im_1_4.at<float>(1,cols-1)+im_1_4.at<float>(0,cols-2);
        im_blur.at<float>(rows-1,cols-1)=im_1_2.at<float>(rows-1,cols-1)+im_1_4.at<float>(rows-2,cols-1)+im_1_4.at<float>(rows-1,cols-2);

        // 3. Pyramid
        if (i!=scales)
        {
            cv::resize(im_blur, v_pyramids[i+1], cv::Size(), scale_fact, scale_fact, CV_INTER_LINEAR);
        }

    }

    SetPose(Eigen::MatrixXf::Identity(4,4));
    // l_points.resize(2000); // TODO reserve space for list
}

Frame::~Frame(){

}

void Frame::Align(const Frame* _prev_fr)
{

}

void Frame::GetPyrLevel(const unsigned int _lev, cv::Mat &_im)
{
    std::unique_lock<std::mutex> lock(m_im); // TODO: Pyramids and image are only read
    if(_lev>scales)
    {
        std::cout << "Scale out of limits!! (GetGradLevel)" << std::endl;
        return;
    }

    _im=v_pyramids[_lev].clone();
}

void Frame::GetGradLevel(const unsigned int _lev, bool _b_X, cv::Mat &_im)
{
    std::unique_lock<std::mutex> lock(m_im); // TODO: Pyramids and image are only read
    if(_lev>scales)
    {
        std::cout << "Scale out of limits!! (GetGradLevel)" << std::endl;
        return;
    }

    if(_b_X)
        _im=v_gradX[_lev].clone();
    else
        _im=v_gradY[_lev].clone();
}

void Frame::ExtractPoints(const unsigned int _n_points, const unsigned int _grid_rows, const unsigned int _grid_cols, const float _min_grad2)
{
    // TODO: Apply grid and maximun nmber of extracted points
    // dist_neigh supposed to be odd
    const int min_dist = 9; // Region of 9x9 pixels can not be choosen for extraction
    v_extr_points.clear();
    v_extr_points.reserve(_n_points);

    mod_grad = v_gradX[0].mul(v_gradX[0]) + v_gradY[0].mul(v_gradY[0]);

    // accept only float type matrices
    CV_Assert(mod_grad.depth() == CV_32FC1);

    int n_rows = mod_grad.rows;
    int n_cols = mod_grad.cols;
    int n_rows_cont = mod_grad.rows;
    int n_cols_cont = mod_grad.cols;

    // Todo, taking advante of continu
    /*if (mod_grad.isContinuous())
    {
        n_cols_cont *= n_rows_cont;
        n_rows_cont = 1;
    }*/

    float* p;
    const int dist_neigh_2 = dist_neigh*dist_neigh;
    const int dist_neigh__2 = dist_neigh/2;
    const int min_dist__2 = min_dist/2;

    // fast access to cv::Mat using pointer to row
    int count=0;
    for(int i = dist_neigh__2; i < n_rows_cont-dist_neigh__2; ++i)
    {
        std::cout << "i: " << i << std::endl;
        p = mod_grad.ptr<float>(i);
        for (int j = dist_neigh__2; j < n_cols_cont-dist_neigh__2; ++j)
        {
            if(p[j]>=_min_grad2)
            {
                std::cout << "    j: " << j << std::endl;

                std::pair<std::vector<int>, std::vector<float> > _pair;
                _pair.first.resize(2);
                _pair.first[0]=j;
                _pair.first[1]=i;
                _pair.second.resize(dist_neigh_2);

                float* p_grad;
                float* p_im;
                size_t count=0;

                // save neighbor pixel values
                for(int i2=-dist_neigh__2;i2<=dist_neigh__2;i2++)
                {
                    p_im = v_pyramids[0].ptr<float>(i+i2);
                    for(int j2=-dist_neigh__2; j2<=dist_neigh__2; j2++, count++)
                        _pair.second[count]=p_im[j+j2];
                }

                // cancel close points
                const int min_row=std::max(0,i-min_dist__2);
                const int min_col=std::max(0,j-min_dist__2);
                const int max_row=std::min(n_rows-1,i+min_dist__2);
                const int max_col=std::min(n_cols-1,j+min_dist__2);

                // std::cout << "min_row/min_col - max_row/max_col: " << min_row << "/" << min_col << "-" << max_row << "/" << max_col << std::endl;
                for(int i2=min_row;i2<=max_row;i2++)
                {
                    p_grad = mod_grad.ptr<float>(i2);
                    for(int j2=min_col; j2<=max_col; j2++)
                        p_grad[j2]=0.f; // To avoid be choosen in next steps
                }

                std::cout << _pair.first[0] << ", " << _pair.first[1] << ": " << _pair.second.front() << std::endl;
                v_extr_points.push_back(_pair);
            }
        }
    }
}

void Frame::AddPoint(Point* _p_pt)
{
    std::unique_lock<std::mutex> lock(m_points);
    l_points.push_back(_p_pt);
}

/*void Frame::AddExtractedPoints(std::list<Point*> &_l_p_all_pts)
{
    std::unique_lock<std::mutex> lock(m_points);


    const float min_inv_depth = 0.1f;
    const float max_inv_depth = 10.0f;
    const float dif_inv_depth = max_inv_depth - min_inv_depth;
    float inv_depth;
    for(auto it=v_extr_points.begin(); it!=v_extr_points.end(); it++)
    {
        // random inverse depth
        inv_depth = min_inv_depth + dif_inv_depth*( static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        Point* p_point = new Point(this, (*it).first[0], (*it).first[1], inv_depth, (*it).second, std::vector<Frame*>());
        // Add to frame points list
        l_points.push_back(p_point);
        // Add to SLAM points list
        _l_p_all_pts.push_back(p_point);
    }
    v_extr_points.clear();
}*/

void Frame::GetPointProjections(std::vector<cv::Point2f> &_v_pt_proj, std::vector<float> &_v_inv_depths)
{
    // Get projection of seen points in this frame

    std::unique_lock<std::mutex> lock(m_points);

    std::cout << "host id: " << id << std::endl;

    _v_pt_proj.resize(l_points.size());
    _v_inv_depths.resize(l_points.size());
    size_t i=0;
    Eigen::Vector2f uv_c;
    float inv_depth;
    for(auto it=l_points.begin(); it!=l_points.end(); it++, i++)
    {
        (*it)->Project(this,uv_c, inv_depth);
        _v_pt_proj[i]=Maths::EigV2f2cvPt2(uv_c);
        //std::cout << inv_depth << std::endl;
        _v_inv_depths[i]=inv_depth;
    }
}


void Frame::SetPose(const Eigen::Matrix<float,4,4> _T_wc)
{
    std::unique_lock<std::mutex> lock(m_pose);
    T_wc = _T_wc;
    T_cw = Maths::InvSE3(T_wc);
}

Eigen::Matrix<float,4,4> Frame::GetPose()
{
    std::unique_lock<std::mutex> lock(m_pose);
    return T_wc;
}

Eigen::Matrix<float,4,4> Frame::GetInvPose()
{
    std::unique_lock<std::mutex> lock(m_pose);
    return T_cw;
}
