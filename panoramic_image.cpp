/*
CV 2021: Final Projects
Panoramic image construction with feature descriptors (LAB4-a)
Student: Vuk Ilic
Professor: Zanuttigh Pietro
*/

#include "panoramic_utils.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
	// Choose the folder from which you want to import images
	cv::String path("kitchen/*.bmp"); // select only images with specific type
	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(path, fn, true);

	// Read the first image
	cv::Mat img1 = cv::imread(fn[0]);

	// Split image into 3 channels - B, G and R
	Mat bgr[3]; 
	split(img1, bgr);

	// Define an angle value (in degrees) which is half of the FoV of the camera used to take the photos
	int angle = 33;

	// Apply cylindrical projection on each channel
	bgr[0] = PanoramicUtils::cylindricalProj(bgr[0], angle);
	bgr[1] = PanoramicUtils::cylindricalProj(bgr[1], angle);
	bgr[2] = PanoramicUtils::cylindricalProj(bgr[2], angle);

	// Merge 3 channels into one image
	merge(bgr, 3, img1);
	
	Mat pan_img = img1;

	// Loop through the rest of the images
	for (size_t k = 1; k < fn.size(); ++k)
	{
		// Read the image
		cv::Mat img2 = cv::imread(fn[k]);
		if (img2.empty()) continue;

		// Split image into 3 channels - B, G and R
		split(img2, bgr);

		// Apply cyplindrical projection on each channel
		bgr[0] = PanoramicUtils::cylindricalProj(bgr[0], angle);
		bgr[1] = PanoramicUtils::cylindricalProj(bgr[1], angle);
		bgr[2] = PanoramicUtils::cylindricalProj(bgr[2], angle);

		// Merge 3 channels into one image
		merge(bgr, 3, img2);

		// Extract the piece of the second image that is not in the first image based on the estimated translation
		// Choose between one of the 2 functions: panorama_orb() or panorama_sift() based on the desired feature extraction methods
		Mat piece = panorama_sift(img1, img2);
		img1 = img2;

		// Panoramic image is created by horizontal concatenation of current panoramic image and extracted piece of new image
		hconcat(pan_img, piece, pan_img);
	}

	// Save and show the panoramic image
	imwrite("test2.jpg", pan_img);
	imshow("Panorama", pan_img);

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

cv::Mat PanoramicUtils::cylindricalProj(
	/* This function projects the image on a cylinder
	If you want to use it on color images, you have to apply it on each channel separately
	*/
	const cv::Mat& image,
	const double angle)
{
	cv::Mat tmp, result;
	tmp = image;
	result = tmp.clone();

	double alpha(angle / 180 * CV_PI);
	double d((image.cols / 2.0) / tan(alpha));
	double r(d / cos(alpha));
	double d_by_r(d / r);
	int half_height_image(image.rows / 2);
	int half_width_image(image.cols / 2);

	for (int x = -half_width_image + 1,
		x_end = half_width_image; x < x_end; ++x)
	{
		for (int y = -half_height_image + 1,
			y_end = half_height_image; y < y_end; ++y)
		{
			double x1(d * tan(x / r));
			double y1(y * d_by_r / cos(x / r));

			if (x1 < half_width_image &&
				x1 > -half_width_image + 1 &&
				y1 < half_height_image &&
				y1 > -half_height_image + 1)
			{
				result.at<uchar>(y + half_height_image, x + half_width_image)
					= tmp.at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
			}
		}
	}

	return result;
}

Mat panorama_sift(Mat img1_color, Mat img2_color) {
	/* This function receives two adjacent color images, estimates the translation between those images based on the sift
	feature matching and as a results it returns the part of the second image that comes after the first image.
	*/
	// Convert color images to grayscale
	Mat img1, img2;
	cvtColor(img1_color, img1, COLOR_BGR2GRAY);
	cvtColor(img2_color, img2, COLOR_BGR2GRAY);

	// Detect the keypoints using SIFT Detector, compute the descriptors
	Ptr<SIFT> detector = SIFT::create();
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	// Matching descriptor vectors with a FLANN based matcher
	// Since SIFT is a floating-point descriptor NORM_L2 is used
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2); // n = 2 means that for every feature we will find 2 best matches

	// Filter matches using the Lowe's ratio test
	// The point of this test is that if the best feature match is not much better than second best, then we reject it
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	// Localize the overlaping part of 2 adjacent images
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	// Calculate 3x3 homography matrix
	Mat H = findHomography(obj, scene, RANSAC);

	// Because of the cylindrical projection relation between 2 adjacent images is a simple translation
	// That's why we only need parameter on the position (0, 2) of the homography matrix - horizontal translation
	int pos = (int)H.at<double>(0, 2);

	// Extract the part of the second image that is not in the first image based on the estimated translation
	int w = img2.cols, h = img2.rows;
	Mat piece = img2_color(Range(0, h), Range(w + pos, w));

	return piece;
}

Mat panorama_orb(Mat img1_color, Mat img2_color) {
	/* This function receives two adjacent color images, estimates the translation between those images based on the orb
	feature matching and as a results it returns the part of the second image that comes after the first image.
	*/
	// Convert color images to grayscale
	Mat img1, img2;
	cvtColor(img1_color, img1, COLOR_BGR2GRAY);
	cvtColor(img2_color, img2, COLOR_BGR2GRAY);

	// // Detect the keypoints using ORB Detector, compute the descriptors
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Ptr<FeatureDetector> detector = ORB::create(500);
	Ptr<DescriptorExtractor> descriptor = ORB::create(500);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(img1, keypoints1);
	descriptor->compute(img1, keypoints1, descriptors1);

	detector->detect(img2, keypoints2);
	descriptor->compute(img2, keypoints2, descriptors2);

	// Matching descriptor vectors with a Brute Force matcher
	// Since ORB is a binary descriptor NORM_HAMMING is used
	vector<DMatch>matches;
	matcher->match(descriptors1, descriptors2, matches);

	// Find the maximum distance and minimum distance of the matching points, which are used to filter the matching points later.
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance < m2.distance; })->distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance < m2.distance; })->distance;

	std::vector< DMatch > good_matches;

	// Filter the matching points according to the minimum distance
	// When the distance between the descriptions is greater than twice the min_dist, it is considered that the matching 
	// is wrong and discarded. But sometimes the minimum distance is very small, such as close to 0, so this will result 
	// in few matches between min_dist and 2*min_dist. So, when 2*min_dist is less than 30, take 30 as the upper limit, 
	// which is less than 30, and the value of 2*min_dist is not used
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}

	// Localize the overlaping part of 2 adjacent images
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	// Calculate 3x3 homography matrix
	Mat H = findHomography(obj, scene, RANSAC);

	// Because of the cylindrical projection relation between 2 adjacent images is a simple translation
	// That's why we only need parameter on the position (0, 2) of the homography matrix - horizontal translation
	int pos = (int)H.at<double>(0, 2);

	// Extract the part of the second image that is not in the first image based on the estimated translation
	int w = img2.cols, h = img2.rows;
	Mat piece = img2_color(Range(0, h), Range(w + pos, w));

	return piece;
}