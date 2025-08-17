#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/flann.hpp>
#include "external/json/json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>

using json = nlohmann::json;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Vec3d rotationMatrixToEulerAngles(Mat &R) {
    double sy = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2,1), R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3d(x, y, z);
}

Mat draw_lines(Mat img, vector<Vec3f> lines, vector<Point2f> pts) {
    Mat img_color;
    cvtColor(img, img_color, COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec3f r = lines[i];
        Point2f pt = pts[i];
        Point pt1(0, -r[2]/r[1]);
        Point pt2(img.cols, -(r[2] + r[0]*img.cols)/r[1]);
        Scalar color(rand() % 255, rand() % 255, rand() % 255);
        line(img_color, pt1, pt2, color, 1);
        circle(img_color, pt, 5, color, -1);
    }
    return img_color;
}

int main() {
    std::ifstream configFile("../config.json");
    if (!configFile) {
        cerr << "Could not open config.json" << endl;
        return -1;
    }

    json config;
    configFile >> config;

    auto K1_data = config["camera_matrices"]["K1"];
    auto K2_data = config["camera_matrices"]["K2"];
    Mat K1(3, 3, CV_64F);
    Mat K2(3, 3, CV_64F);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            K1.at<double>(i, j) = K1_data[i][j];
            K2.at<double>(i, j) = K2_data[i][j];
        }

    int table_number = config["lsh_params"]["table_number"];
    int key_size = config["lsh_params"]["key_size"];
    int multi_probe_level = config["lsh_params"]["multi_probe_level"];

    Mat img1 = imread("../imgs_kiti/scene-1/F.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("../imgs_kiti/scene-1/L.jpg", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Error reading images." << endl;
        return -1;
    }

    map<string, Ptr<Feature2D>> detectors = {
        {"ORB", ORB::create()},
        {"AKAZE", AKAZE::create()},
        {"BRISK", BRISK::create()},
        {"KAZE", KAZE::create()}
    };

    try {
        detectors["SIFT"] = SIFT::create();
    } catch (...) {
        cout << "SIFT not available." << endl;
    }

    FastFeatureDetector::DetectorType fastType = FastFeatureDetector::TYPE_9_16;
    Ptr<FeatureDetector> fast = FastFeatureDetector::create(10, true, fastType);
    Ptr<DescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();
    detectors["FAST+BRIEF"] = nullptr;

    for (auto &[name, detector] : detectors) {
        cout << "\n--- Using " << name << " ---" << endl;

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        if (name == "FAST+BRIEF") {
            fast->detect(img1, kp1);
            fast->detect(img2, kp2);
            brief->compute(img1, kp1, des1);
            brief->compute(img2, kp2, des2);
        } else {
            detector->detectAndCompute(img1, noArray(), kp1, des1);
            detector->detectAndCompute(img2, noArray(), kp2, des2);
        }

        if (des1.empty() || des2.empty()) {
            cout << "Descriptors could not be computed." << endl;
            continue;
        }

        Ptr<DescriptorMatcher> matcher;

        if (des1.type() == CV_32F || name == "SIFT") {
            matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        } else {
            if (des1.type() != CV_8U) {
                des1.convertTo(des1, CV_8U);
                des2.convertTo(des2, CV_8U);
            }
            auto indexParams = makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);
            auto searchParams = makePtr<cv::flann::SearchParams>();
            matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
        }

        vector<vector<DMatch>> knn_matches;
        matcher->knnMatch(des1, des2, knn_matches, 2);

        vector<DMatch> good_matches;
        for (auto &m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < 0.75 * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        cout << "Good matches: " << good_matches.size() << endl;

        if (good_matches.size() < 8) {
            cout << "Not enough matches to compute fundamental matrix." << endl;
            continue;
        }

        vector<Point2f> pts1, pts2;
        for (auto &m : good_matches) {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        Mat F, mask;
        F = findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.0, 0.999, mask);

        if (F.empty()) {
            cout << "Could not compute fundamental matrix." << endl;
            continue;
        }

        vector<Point2f> pts1_inliers, pts2_inliers;
        for (int i = 0; i < mask.rows; ++i) {
            if (mask.at<uchar>(i)) {
                pts1_inliers.push_back(pts1[i]);
                pts2_inliers.push_back(pts2[i]);
            }
        }

        Mat E = K2.t() * F * K1;

        Mat R, t, pose_mask;
        recoverPose(E, pts1_inliers, pts2_inliers, K1, R, t, pose_mask);

        cout << "Rotation (R):\n" << R << endl;
        cout << "Translation (t):\n" << t << endl;

        Vec3d euler = rotationMatrixToEulerAngles(R);
        euler *= 180.0 / CV_PI;
        cout << "Euler angles (degrees): Roll: " << euler[0] << ", Pitch: " << euler[1] << ", Yaw: " << euler[2] << endl;

        vector<Vec3f> lines1, lines2;
        computeCorrespondEpilines(pts2_inliers, 2, F, lines1);
        computeCorrespondEpilines(pts1_inliers, 1, F, lines2);

        Mat img1_lines = draw_lines(img1, lines1, pts1_inliers);
        Mat img2_lines = draw_lines(img2, lines2, pts2_inliers);

        imshow(name + " - Epipolar Lines 1", img1_lines);
        imshow(name + " - Epipolar Lines 2", img2_lines);
        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}