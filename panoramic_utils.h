// Computer Vision 2021 (P. Zanuttigh) - LAB 4 

#ifndef LAB4__PANORAMIC__UTILS__H
#define LAB4__PANORAMIC__UTILS__H

#include <memory>
#include <iostream>



#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


class PanoramicUtils
{
public:
    static
        cv::Mat cylindricalProj(
            const cv::Mat& image,
            const double angle);
 
};

Mat panorama_orb(Mat img1_color, Mat img2_color);

Mat panorama_sift(Mat img1_color, Mat img2_color);

#endif // LAB4__PANORAMIC__UTILS__H
