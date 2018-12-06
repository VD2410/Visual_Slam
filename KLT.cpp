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
//  detector->set("thres", 100);
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
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset){
    Matx33d F;
    Mat A(prev_subset.size(), 9, CV_64FC1);
    Mat Point2(3, 1, CV_64FC1);
    Mat Point1(3, 1, CV_64FC1);

    Mat Norm(3,3, CV_64FC1);
    Norm.at<double>(0,0) = 2.0/640.0;
    Norm.at<double>(0,1)=0;
    Norm.at<double>(0,2)=-1;
    Norm.at<double>(1,0)=0;
    Norm.at<double>(1,1)=2.0/480.0;
    Norm.at<double>(1,2)=-1;
    Norm.at<double>(2,0)=0;
    Norm.at<double>(2,1)=0;
    Norm.at<double>(2,2)=1;

    for (size_t i=0; i<prev_subset.size(); i++)
    {
        Point1.at<double>(0,0)=prev_subset[i].x;
        Point1.at<double>(1,0)=prev_subset[i].y;
        Point1.at<double>(2,0)=1.0;
    Point1 = Norm * Point1;
        Point2.at<double>(0,0)=next_subset[i].x;
        Point2.at<double>(1,0)=next_subset[i].y;
        Point2.at<double>(2,0)=1.0;
        Point2 = Norm * Point2;
        Mat Point = Point2 * Point1.t();
        A.at<double>(i,0)=Point.at<double>(0,0);
        A.at<double>(i,1)=Point.at<double>(0,1);
        A.at<double>(i,2)=Point.at<double>(0,2);
        A.at<double>(i,3)=Point.at<double>(1,0);
        A.at<double>(i,4)=Point.at<double>(1,1);
        A.at<double>(i,5)=Point.at<double>(1,2);
        A.at<double>(i,6)=Point.at<double>(2,0);
        A.at<double>(i,7)=Point.at<double>(2,1);
        A.at<double>(i,8)=Point.at<double>(2,2);

    }

    SVD svd(A);
    cv::Mat vt(9, 1, CV_64FC1);

    cv::Mat f_test(3,3, CV_64FC1);
    f_test.at<double>(0,0)=svd.vt.at<double>(svd.vt.rows-1,0);
    f_test.at<double>(0,1)=svd.vt.at<double>(svd.vt.rows-1,1);
    f_test.at<double>(0,2)=svd.vt.at<double>(svd.vt.rows-1,2);
    f_test.at<double>(1,0)=svd.vt.at<double>(svd.vt.rows-1,3);
    f_test.at<double>(1,1)=svd.vt.at<double>(svd.vt.rows-1,4);
    f_test.at<double>(1,2)=svd.vt.at<double>(svd.vt.rows-1,5);
    f_test.at<double>(2,0)=svd.vt.at<double>(svd.vt.rows-1,6);
    f_test.at<double>(2,1)=svd.vt.at<double>(svd.vt.rows-1,7);
    f_test.at<double>(2,2)=svd.vt.at<double>(svd.vt.rows-1,8);
    

    SVD svd_F(f_test); 
    Mat rank2 = Mat::zeros(3,3, CV_64FC1);
    rank2.at<double>(0,0) = svd_F.w.at<double>(0,0);
    rank2.at<double>(1,1) = svd_F.w.at<double>(1,0);
    f_test = svd_F.u * rank2 * svd_F.vt;
    f_test = (Norm.t()*f_test*Norm);

    F= f_test;

    // F(0,0) = F(0,0) / F(2,2);
    // F(0,1) = F(0,1) / F(2,2);
    // F(0,2) = F(0,2) / F(2,2);
    // F(1,0) = F(1,0) / F(2,2);
    // F(1,1) = F(1,1) / F(2,2);
    // F(2,0) = F(2,0) / F(2,2);
    // F(2,1) = F(2,1) / F(2,2);
    // F(2,2) = F(2,2) / F(2,2);
    // F(1,2) = F(1,2) / F(2,2);

    return F;
}



bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d candidate,double threshold){
    
    Matx31d Point1;
    Point1(0,0)=prev_keypoint.x;
    Point1(0,1)=prev_keypoint.y;
    Point1(0,2)=1.0;
    Matx31d Point2;
    Point2(0,0)=next_keypoint.x;
    Point2(0,1)=next_keypoint.y;
    Point2(0,2)=1.0;
    auto epipolar_line = candidate.t()*Point2;
    float a = epipolar_line(0,0);
    float b = epipolar_line(1,0);
    float c = epipolar_line(2,0);
    float u = Point1(0,0);
    float v = Point1(0,1);
    float distance = abs(a*u+b*v+c)/sqrt(a*a+b*b);
    
    if(distance<=threshold)
        return true;
    else
        return false;
}

void displayEpipolar(Mat img_1, Mat img_2, vector<cv::Point2f> prev_keypoints,vector<cv::Point2f> next_keypoints, Matx33d f)
{
   // visualize all  keypoints
   hconcat(img_1,img_2,img_1);
   for ( size_t i=0; i< prev_keypoints.size() ;i++)
   {
           Matx31d Point2;
            Point2(0,0)=next_keypoints[i].x;
            Point2(0,1)=next_keypoints[i].y;
            Point2(0,2)=1.0;
            auto epipolar_line = f.t()*Point2;
            double a = epipolar_line(0,0);
            double b = epipolar_line(1,0);
            double c = epipolar_line(2,0);
            Point pt1, pt2;
            pt1.x =img_2.size[1];
            pt1.y =-c/b;
            pt2.x = img_2.size[1]+img_2.size[1];
            pt2.y = -(c+a*img_2.size[1])/b;
            line(img_1, pt1, pt2, cv::Scalar(0,255,255));
            circle(img_1, Point(prev_keypoints[i].x+img_2.size[1], prev_keypoints[i].y), 4, cv::Scalar(0,100,250),CV_FILLED);

            Matx31d Point1;
            Point1(0,0)=prev_keypoints[i].x;
            Point1(0,1)=prev_keypoints[i].y;
            Point1(0,2)=1.0;
            epipolar_line = f*Point1;
            a = epipolar_line(0,0);
            b = epipolar_line(1,0);
            c = epipolar_line(2,0);
            Point pt3, pt4;
            pt3.x =0;
            pt3.y =-c/b;
            pt4.x = img_2.size[1];
            pt4.y = -(c+a*img_2.size[1])/b;

           line(img_1, pt3, pt4, cv::Scalar(160,0,255));
           circle(img_1, next_keypoints[i], 4, cv::Scalar(178,0,100),CV_FILLED);
       
   }

   cv::imshow("klt tracker", img_1);
   cv::waitKey(0);

}

void findPose(const Matx33d &E, Matx44d &P1, Matx44d &P2, Matx44d &P3, Matx44d &P4)
{       Matx33d W(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        Matx33d Z(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
        SVD svd(E);
        Mat R1 = svd.u * Mat(W).t() * svd.vt;
        if(cv::determinant(R1)<0)
            R1=-R1;

         Mat R2 = svd.u * Mat(W) * svd.vt;
        if(cv::determinant(R2)<0)
            R2=-R2;

        double scal = (svd.w.at<double>(0,0)+svd.w.at<double>(1,0))/2; 
        cout<<"svd(E) w"<<endl<<svd.w<<endl;
        Mat S1 = -svd.u * Mat(Z) * svd.u.t();
        Mat S2 = svd.u * Mat(Z) * svd.u.t();
        S1 = scal*S1;
        S2 = scal*S2;
        
        SVD svd_S1(S1);
        Mat u3_1(3, 1, CV_64FC1);
        u3_1.at<double>(0,0) = svd_S1.vt.at<double>(2,0);
        u3_1.at<double>(1,0) = svd_S1.vt.at<double>(2,1);
        u3_1.at<double>(2,0) = svd_S1.vt.at<double>(2,2);
        u3_1 = u3_1;

        SVD svd_S2(S2);
        Mat u3_2(3, 1, CV_64FC1);
        u3_2.at<double>(0,0) = svd_S2.vt.at<double>(2,0);
        u3_2.at<double>(1,0) = svd_S2.vt.at<double>(2,1);
        u3_2.at<double>(2,0) = svd_S2.vt.at<double>(2,2);
        u3_2 = u3_2;
        for (int i=0; i<3; i++)
        {
            for(int j = 0; j<3; j++)
                P1(i,j) = R2.at<double>(i,j);
            P1(i,3) = u3_2.at<double>(i,0);
        } 


        for (int i=0; i<3; i++)
        {
            for(int j = 0; j<3; j++)
                P2(i,j) = R1.at<double>(i,j);
            P2(i,3) = u3_1.at<double>(i,0);
        } 
        for (int i=0; i<3; i++)
        {
            for(int j = 0; j<3; j++)
                P3(i,j) = R2.at<double>(i,j);
            P3(i,3) = -u3_2.at<double>(i,0);
        }

        for (int i=0; i<3; i++)
        {
            for(int j = 0; j<3; j++)
                P4(i,j) = R1.at<double>(i,j);
            P4(i,3) = -u3_1.at<double>(i,0);
        }

        cout<<"P1"<<P1<<endl<<"P2"<<endl<<P2<<endl<<"P3"<<P3<<endl<<"P4"<<P4;
   
}


int triangulated_3d_map_points(Matx44d P, vector<cv::Point2f> prev_point, vector<cv::Point2f> next_point , Matx33d K)
{
    Matx34d Projection_Matrix(1, 0, 0, 0, 0 ,1, 0, 0, 0, 0, 1, 0);
    Matx44d R(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx44d Translation(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx34d P1 = K*Projection_Matrix*R*Translation;

    int count=0;

    for(size_t i=0; i<prev_point.size(); i++)
    {
        Mat A(4,4,CV_64FC1) ;
        A.row(0) = prev_point[i].x*Mat(P1.row(2)) - Mat(P1.row(0));
        A.row(1) = prev_point[i].y*Mat(P1.row(2)) - Mat(P1.row(1));
        A.row(2) = next_point[i].x*Mat((K*Projection_Matrix*P).row(2)) - Mat((K*Projection_Matrix*P).row(0));
        A.row(3) = next_point[i].y*Mat((K*Projection_Matrix*P).row(2)) - Mat((K*Projection_Matrix*P).row(1));

        SVD svd(A);


        double a = svd.vt.at<double>(3,0);
        double b = svd.vt.at<double>(3,1);
        double c = svd.vt.at<double>(3,2);
        double d = svd.vt.at<double>(3,3);

        Matx41d x_w(a, b, c, d);
        Matx41d x_c = P * x_w;
        if((c/d)>0 && (x_c(0,2)/x_c(0,3))>0)
        {
            count++;
        }

        }
    return count;
}

Vector<cv::Point3f> triangulationPoints(Matx44d P, vector<cv::Point2f> prev_point, vector<cv::Point2f> next_point , Matx33d K)
{
    Vector <cv::Point3f> tringulation_point;
    Matx34d Projection_Matrix(1, 0, 0, 0, 0 ,1, 0, 0, 0, 0, 1, 0);
    Matx44d R(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx44d Translation(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    Matx34d P1 = K*Projection_Matrix*R*Translation;
    for(size_t i=0; i<prev_point.size(); i++)
    {
        Mat A(4,4,CV_64FC1) ;
        A.row(0) = prev_point[i].x*Mat(P1.row(2)) - Mat(P1.row(0));
        A.row(1) = prev_point[i].y*Mat(P1.row(2)) - Mat(P1.row(1));
        A.row(2) = next_point[i].x*Mat((K*Projection_Matrix*P).row(2)) - Mat((K*Projection_Matrix*P).row(0));
        A.row(3) = next_point[i].y*Mat((K*Projection_Matrix*P).row(2)) - Mat((K*Projection_Matrix*P).row(1));
        
        SVD svd(A);

        double a = svd.vt.at<double>(3,0);
        double b = svd.vt.at<double>(3,1);
        double c = svd.vt.at<double>(3,2);
        double d = svd.vt.at<double>(3,3);
        tringulation_point.push_back(Point3f(a/d, b/d, c/d));
    }
    return tringulation_point;
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
    cv::Matx33d F,candidate;
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
        candidate = Findfundamental(prev_subset,next_subset);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<kps_prev.size();j++){
            if(checkinlier(kps_prev[j],kps_next[j],candidate,d))
                inliers++;
        }
        if(inliers > bestinliers)
        {
            F = candidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.
    for(size_t j=0;j<kps_prev.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    F = Findfundamental(prev_subset,next_subset);
    
    cout<<"Fundamental matrix-----------------------------------"<<endl<<F<<endl;
    displayEpipolar(img_1, img_2, prev_subset, next_subset, F);
    FileStorage fs("../config/default.yaml", FileStorage::READ);
    double fx = fs["camera.fx"];
    double fy = fs["camera.fy"];
    double cx = fs["camera.cx"];
    double cy = fs["camera.cy"];
    Matx33d K(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    Matx33d E = K.t() * F * K;
    Matx44d P1, P2, P3, P4, bestPose;
    findPose(E, P1, P2, P3, P4);
    
    vector<Matx44d> poseList = {P1, P2, P3, P4};
    int poseInliers = 0;

    for(int i = 0; i<4; i++)
    {
        int Inliners = triangulated_3d_map_points(poseList[i], prev_subset, next_subset, K);
        if(Inliners > poseInliers)
        {
            poseInliers=Inliners;
            bestPose=poseList[i];
            cout<<Inliners<<endl;
        }
    }

    cout<<"Essential matrix"<<endl<<E<<endl;
    cout<<"R & T"<<endl<<bestPose<<endl;
    Matx34d Projection_Matrix(1, 0, 0, 0, 0 ,1, 0, 0, 0, 0, 1, 0);
    Matx34d bestCameraPose = K*Projection_Matrix*bestPose;
    cout<<"Best Camera Pose "<<endl<<bestCameraPose<<endl;
    Vector<Point3f> test = triangulationPoints(bestPose, prev_subset, next_subset, K);
    cout<<"Tringulation point"<<endl<<test[0]<<endl;

    return 0;
}