////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}


//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset){
    cv::Matx33d F;
    cv::Mat A(1,9,CV_64FC1);
    cv::Mat f(3,3,CV_64FC1);
    cv::Mat Normalization(3,3,CV_64FC1,double(0));
    Normalization.at<double>(0,0) = 2.0 / 640.0;
    Normalization.at<double>(0,2) = -1;
    Normalization.at<double>(1,1) = 2.0 / 480.0;
    Normalization.at<double>(1,2) = -1;
    Normalization.at<double>(2,2) = 1;
    
    cv::Mat Point1(3,1,CV_64FC1);
    cv::Mat Point2(3,1,CV_64FC1);
    for(int i = 0; i < prev_subset.size(); i++){
        Point1.at<double>(0,0) = prev_subset[i].x;
        Point1.at<double>(1,0) = prev_subset[i].y;
        Point1.at<double>(2.0) = 1;
        
        Point2.at<double>(0,0) = next_subset[i].x;
        Point2.at<double>(1,0) = next_subset[i].y;
        Point2.at<double>(2.0) = 1;
        
        Point1 *= Normalization;
        Point2 *= Normalization;
        cv:Mat Point = Point2 * Point1.t();
        int q = 0;
        for(int j = 0; j<3;j++){
            for(int k = 0; k<3;k++){
                A.at<double>(i,q) = Point.at<double>(j,k);
                q++;
            }
        }
    }
        
    SVD svd(A);
        
    q = 0;
    for(int j = 0; j<3;j++){
        for(int k = 0; k<3;k++){
            f.at<double>(j,k) = svd.vt.at<double>(svd.vt.rows-1,q);
            q++;
        }
    }
        
    SVD newsvd(f);
    cv::Mat w(3,3,CV_64FC1,double(0));
        
    w.at<double>(0,0) = newsvd.w.at<double>(0,0);
    w.at<double>(1,1) = newsvd.w.at<double>(1,0);
    F = newsvd.u * w * newsvd.vt;
    F = Normalization.t() * F * Normalization;
        
    //for(int j = 0; j<3;j++){
    //     for(int k = 0; k<3;k++){
    //        F.at<double>(j,k) /= F.at<double>(2,2);
    //     }
    //}
        
        
    
    
    //fill the blank
    return F;
}
bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    //fill the blank
    cv::Mat Point1(3,1,CV_64FC1,double(1));
    cv::Mat Point2(3,1,CV_64FC1,double(1));
        
    Point2.at<double>(0,0) = next_subset.x;
    Point2.at<double>(1,0) = next_subset.y;
    
    Mat epiLine = Fcandidate.t() * Point2;
    
    double a = epiLine(0,0);
    double b = epiLine(1,0);
    double c = epiLine(2,0);
    
    Point1.at<double>(0,0) = prev_subset.x;
    Point1.at<double>(1,0) = prev_subset.y;
    
    double u = Point1.at<double>(0,0);
    double v = Point1.at<double>(1,0);
    
    double distance = abs(a*u+b*v+c)/sqrt(a*a+b*b);
    
    if(distance<=d)
        return true;
    
    return false;
}




int main( int argc, char** argv )
{

    srand ( time(NULL) );

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);


    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    vector<cv::Point2f> kps_prev,kps_next;
    kps_prev.clear();
    kps_next.clear();
    for(size_t i=0;i<prev_keypoints.size();i++)
    {
        if(status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }


    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0-p)/std::log(1.0-std::pow(1.0-e,8))));
    Mat Fundamental;
    cv::Matx33d F,Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset,next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for(int i=0;i<niter;i++){
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while(rand_util.size()<8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());
        for(size_t j = 0;j<rand_util.size();j++){
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F
        Fcandidate = Findfundamental(prev_subset,next_subset);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<prev_keypoints.size();j++){
            if(checkinlier(prev_keypoints[j],next_keypoints[j],Fcandidate,d))
                inliers++;
        }
        if(inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.

    for(size_t j=0;j<prev_keypoints.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    F = Findfundamental(prev_subset,next_subset);

    cout<<"Fundamental matrix is \n"<<F<<endl;
    
    //Visualize
    hconcat(img_1,img_2,img_1);
    for ( int i=0; i< prev_subset.size() ;i++)
    {
        
        cv::Mat Point1(3,1,CV_64FC1,double(1));
        cv::Mat Point2(3,1,CV_64FC1,double(1));
        
        Point2.at<double>(0,0) = next_subset[i].x;
        Point2.at<double>(1,0) = next_subset[i].y;
        
        
        Point1.at<double>(0,0) = prev_subset[i].x;
        Point1.at<double>(1,0) = prev_subset[i].y;
        
        Point pt1,pt2;
        cv::Mat epiLine = Fcandidate.t()*Point2;
        pt2.x =  next_subset[i].x + img_2.size[1];
        pt2.y =  -((epiLine.at<double>(2,0)+epiLine.at<double>(0,0)*img_2.size[1])/epiLine.at<double>(1,0));
        
        pt1.x = img_2.size[1];
        pt1.y = -(epiLine.at<double>(2,0)/epiLine.at<double>(1,0))

        line(img_1, pt1, pt2, cv::Scalar(0,255,255));
        circle(img_1, next_subset[i], 5, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);
    
    //Triangulation
    cv::Mat K(3,3,CV_64FC1,double(0));
    cv::FileStorage file = cv::FileStorage( "../config/default.yaml", cv::FileStorage::READ );
    double fx,fy,cx,cy;
    fx = file["camera.fx"];
    fy = file["camera.fy"];
    cx = file["camera.cx"];
    cy = file["camera.cy"]; 
    
    K.at<double>(0,0) = fx;
    K.at<double>(0,2) = cx;
    K.at<double>(1,1) = fy;
    K.at<double>(1,2) = cx;
    K.at<double>(2,2) = 1;
    
    cv::Mat E_mat(3,3,CV_64FC1,double(0));
    
    E_mat = K.t()*F*K;
    
    cout<<"Essential matrix is \n"<<E_mat<<endl;
    
    return 0;
}
