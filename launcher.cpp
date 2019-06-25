#include <Eigen/Dense>

#include <iterator>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include "slam.h"
#include "point.h"






int main(int argc, char *argv[])
{
    // std::string path_to_data = "/home/carlos/Datasets/dataset_1/cam0";
    // std::string path_to_calibration = "/home/carlos/Datasets/dataset_1/EuRoC.yaml";

    std::string path_to_data = "/home/ccampos/DiVisO/dataset/cam0/";
    std::string path_to_calibration = "/home/ccampos/DiVisO/dataset/EuRoC.yaml";

    std::cout << "DiVisO SLAM" << std::endl;

    SLAM DiVisO(path_to_data, path_to_calibration);
    DiVisO.Run();

    return 0;
}

