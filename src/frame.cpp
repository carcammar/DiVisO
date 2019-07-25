#include "frame.h"
#include "point.h"

Frame::Frame(unsigned int _id, const std::string _path_to_image, const unsigned int _scales, const double _scale_fact, const int _dist_negih, Camera* _p_cam):
    id(_id), scales(_scales), scale_fact(_scale_fact), dist_neigh(_dist_negih), b_KF(false), p_cam(_p_cam)
{
    // TODO: Parallelize this method
    std::unique_lock<std::mutex> lock(m_im);

    im_8u = cv::imread(_path_to_image, CV_LOAD_IMAGE_GRAYSCALE);
    im_8u.convertTo(im, CV_32F);

    if (im.empty())
        std::cout << "Error: EMPTY IMAGE" << std::endl;

    // TODO: ideally remove undistortion
    std::cout << "Cam matrix: " << p_cam->GetKcv() << std::endl;
    std::cout << "Dist parameters: " << p_cam->GetDist() << std::endl;
    std::cout << "im size: " << im.size() << std::endl;
    cv::undistort(im.clone(), im, p_cam->GetKcv(), p_cam->GetDist());

    v_pyramids.resize(scales+1);
    v_scale_fact.resize(scales+1);
    v_inv_scale_fact.resize(scales+1);
    v_gradX.resize(scales+1);
    v_gradY.resize(scales+1);


    v_pyramids[0]=im.clone(); // TODO: clone?
    v_scale_fact[0] << 0.f, 0.f;
    v_inv_scale_fact[0] << 0.f, 0.f;

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
            v_scale_fact[i+1] << static_cast<float>(v_pyramids[i+1].rows)/static_cast<float>(v_pyramids[i].rows), static_cast<float>(v_pyramids[i+1].cols)/static_cast<float>(v_pyramids[i].cols);
            v_inv_scale_fact[i+1] << 1.f/v_scale_fact[i+1](0), 1.f/v_scale_fact[i+1](1);
        }

    }

    SetPose(Eigen::MatrixXf::Identity(4,4));
    // l_points.resize(2000); // TODO reserve space for list
}

Frame::Frame(Frame* _p_fr): id(_p_fr->id), b_KF(_p_fr->b_KF), p_cam(_p_fr->p_cam), scales(_p_fr->scales), scale_fact(_p_fr->scale_fact),
    dist_neigh(_p_fr->dist_neigh), l_points(_p_fr->l_points), T_wc(_p_fr->T_wc), T_cw(_p_fr->T_cw)
{
    im = _p_fr->im.clone();
    im_8u = _p_fr->im_8u.clone();
    im_und = _p_fr->im_und.clone();
    feasible_pts = _p_fr->feasible_pts.clone();
    mod_grad = _p_fr->mod_grad.clone();
    v_extr_points = _p_fr->v_extr_points;

    v_pyramids.resize(_p_fr->v_pyramids.size());
    v_scale_fact.resize(_p_fr->v_pyramids.size());
    v_inv_scale_fact.resize(_p_fr->v_pyramids.size());
    v_gradX.resize(_p_fr->v_gradX.size());
    v_gradY.resize(_p_fr->v_gradY.size());

    for (size_t i = 0; i<_p_fr->v_pyramids.size(); i++)
    {
        v_scale_fact[i] = _p_fr->v_scale_fact[i];
        v_inv_scale_fact[i] = _p_fr->v_inv_scale_fact[i];
        v_pyramids[i] = _p_fr->v_pyramids[i].clone();
        v_gradX[i] = _p_fr->v_gradX[i].clone();
        v_gradY[i]= _p_fr->v_gradY[i].clone();
    }
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

float Frame::GetIntPoint(const Eigen::Vector2f &_uv, const unsigned int _lev)
{
    // TODO: maybe faster with cv::Mat operations
    int x_0 = static_cast<int>(floor(_uv(0)));
    int y_0 = static_cast<int>(floor(_uv(1)));
    int x_1 = x_0+1;
    int y_1 = y_0+1;

    Eigen::Vector2f aux1, aux2;
    aux1 << static_cast<float>(x_1)-_uv(0) , _uv(0)-static_cast<float>(x_0);
    aux2 << static_cast<float>(y_1)-_uv(1) , _uv(1)-static_cast<float>(y_0);

    std::cout << "v_pyramids[_lev].size() = " << v_pyramids[_lev].size() << "       _uv = " << _uv.transpose() <<  std::endl;
    float *p0 = v_pyramids[_lev].ptr<float>(y_0);
    float *p1 = v_pyramids[_lev].ptr<float>(y_1);
    float i_0_0 = p0[x_0];
    float i_0_1 = p1[x_0];
    float i_1_0 = p0[x_1];
    float i_1_1 = p1[x_1];

    return aux1(0)*(i_0_0*aux2(0)+i_0_1*aux2(1))+aux1(1)*(i_1_0*aux2(0)+i_1_1*aux2(1));
}

void Frame::GetGradPoint(const Eigen::Vector2f &_uv, const unsigned int _lev, Eigen::Vector2f &_grad)
{
    // TODO: maybe faster with cv::Mat operations
    int x_0 = static_cast<int>(floor(_uv(0)));
    int y_0 = static_cast<int>(floor(_uv(1)));
    int x_1 = x_0+1;
    int y_1 = y_0+1;
    // weights
    Eigen::Vector2f aux1, aux2;
    aux1 << static_cast<float>(x_1)-_uv(0) , _uv(0)-static_cast<float>(x_0);
    aux2 << static_cast<float>(y_1)-_uv(1) , _uv(1)-static_cast<float>(y_0);

    float *p0 = v_gradX[_lev].ptr<float>(y_0);
    float *p1 = v_gradX[_lev].ptr<float>(y_1);
    float g_0_0 = p0[x_0];
    float g_0_1 = p1[x_0];
    float g_1_0 = p0[x_1];
    float g_1_1 = p1[x_1];
    // std::cout << "g_0_0=" << g_0_0 << std::endl;
    _grad(0) = aux1(0)*(g_0_0*aux2(0)+g_0_1*aux2(1))+aux1(1)*(g_1_0*aux2(0)+g_1_1*aux2(1));

    p0 = v_gradY[_lev].ptr<float>(y_0);
    p1 = v_gradY[_lev].ptr<float>(y_1);
    g_0_0 = p0[x_0];
    g_0_1 = p1[x_0];
    g_1_0 = p0[x_1];
    g_1_1 = p1[x_1];
    // std::cout << "g_0_0=" << g_0_0 << std::endl;
    _grad(1) = aux1(0)*(g_0_0*aux2(0)+g_0_1*aux2(1))+aux1(1)*(g_1_0*aux2(0)+g_1_1*aux2(1));
}

void Frame::GetScaleFact(const unsigned _lev, Eigen::Vector2f &_scale_fact)
{
    if(_lev>scales)
    {
        std::cout << "Scale out of limits!! (GetScaleFact)" << std::endl;
        return;
    }
    _scale_fact = v_scale_fact[_lev];
}

void Frame::GetScaleFact(std::vector<Eigen::Vector2f> &_v_scale_fact)
{
    _v_scale_fact = v_scale_fact;
}


void Frame::GetScaleSize(const unsigned _lev, Eigen::Vector2i &_scale_size)
{
    if(_lev>scales)
    {
        std::cout << "Scale out of limits!! (GetScaleSize)" << std::endl;
        return;
    }
    _scale_size(0) = v_pyramids[_lev].rows;
    _scale_size(1) = v_pyramids[_lev].cols;
}

void Frame::GetScaleSize(std::vector<Eigen::Vector2i> &_v_scale_size)
{
    _v_scale_size.resize(scales+1);
    for(int i=0; i<=scales; i++)
    {
        _v_scale_size[i](0) = v_pyramids[i].rows;
        _v_scale_size[i](1) = v_pyramids[i].cols;
    }
}


void Frame::ChangeScalePoint(const unsigned int _scale_0, const unsigned int _scale_1, Eigen::Vector2f &_pt)
{
    if (_scale_0==_scale_1)
        return;
    else if(_scale_0>_scale_1)
    {
        for(int lev = _scale_0; lev>_scale_1; lev--)
        {
            _pt(0) *= v_inv_scale_fact[lev](0);
            _pt(1) *= v_inv_scale_fact[lev](1);
        }
    }
    else {
        for(int lev = _scale_0+1; lev<=_scale_1; lev++)
        {
            _pt(0) *= v_scale_fact[lev](0);
            _pt(1) *= v_scale_fact[lev](1);
        }
    }
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

    // Todo, taking advante of continue
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
        // std::cout << "i: " << i << std::endl;
        p = mod_grad.ptr<float>(i);
        for (int j = dist_neigh__2; j < n_cols_cont-dist_neigh__2; ++j)
        {
            if(p[j]>=_min_grad2)
            {
                // std::cout << "    j: " << j << std::endl;

                std::pair<Eigen::Vector2f, std::vector<float> > _pair;
                _pair.first(0)=static_cast<float>(j);
                _pair.first(1)=static_cast<float>(i);
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

                // std::cout << _pair.first[0] << ", " << _pair.first[1] << ": " << _pair.second.front() << std::endl;
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
