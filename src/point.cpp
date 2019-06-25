#include "point.h"

Point::Point(Frame* _p_host_fr, float *_uv_host_fr, float _inv_depth,
             std::vector<float> _v_neigh_pixels, std::vector<Frame*> _v_obs_frs):
    p_host_fr(_p_host_fr), inv_depth(_inv_depth), v_neigh_pixels(_v_neigh_pixels), v_obs_frs(_v_obs_frs)
{
    uv_host_fr[0]=_uv_host_fr[0];
    uv_host_fr[1]=_uv_host_fr[1];
    Eigen::Vector4f h_X = p_host_fr->p_cam->UnprojectInc(uv_host_fr, inv_depth);
    w_xyz = (p_host_fr->GetPose()*h_X).head(3);
    std::cout << "h_X: " << h_X << std::endl;
    std::cout << "p_host_fr->GetPose(): " << p_host_fr->GetPose() << std::endl;
    std::cout << "w_xyz: " << w_xyz << std::endl;
}


void Point::Project(const Eigen::Matrix<float,4,4> _T_cw, Eigen::Vector2f &_uv_c)
{
    // TODO: Add special case when C is host frame to save computational time
    Eigen::Matrix<float,4,4> T_wh = p_host_fr->GetPose();
    //std::cout << T_wh << std::endl;
    Eigen::Vector2f uv_h;

    uv_h[0]=uv_host_fr[0];
    uv_h[1]=uv_host_fr[1];

    Eigen::Vector4f h_X = p_host_fr->p_cam->UnprojectInc(uv_h, inv_depth);
    Eigen::Vector4f c_X = _T_cw*T_wh*h_X;
    //std::cout << c_X << std::endl;
    _uv_c = p_host_fr->p_cam->Project(c_X.head(3)); // TODO: Change for case when camera and host have different calibration
}

void Point::Project(Frame* _p_fr, Eigen::Vector2f &_uv_c, float &_inv_depth)
{
    if (_p_fr==p_host_fr)
    {
        _uv_c << uv_host_fr[0], uv_host_fr[1];
        _inv_depth = inv_depth;
        return;
    }

    // TODO: Add special case when C is host frame to save computational time
    const Eigen::Matrix<float,4,4> _T_cw = _p_fr->GetInvPose();
    Eigen::Matrix<float,4,4> T_wh = p_host_fr->GetPose();
    Eigen::Vector2f uv_h;

    uv_h[0]=uv_host_fr[0];
    uv_h[1]=uv_host_fr[1];

    Eigen::Vector4f h_X = p_host_fr->p_cam->UnprojectInc(uv_h, inv_depth);
    Eigen::Vector4f c_X = _T_cw*T_wh*h_X;
    _inv_depth = 1.f/c_X(2);
    _uv_c = p_host_fr->p_cam->Project(c_X.head(3));
}

void Point::Project(Frame* _p_fr, Eigen::Vector2f &_uv_c)
{
    float _depth;
    Project(_p_fr, _uv_c, _depth);
}

unsigned int Point::GetHostId()
{
    return p_host_fr->id;
}

Eigen::Vector3f Point::GetWorldPos()
{
    // TODO: needed mutex??
    return w_xyz;
}

