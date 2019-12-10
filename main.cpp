#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <chrono>
#include <iomanip>
#include "ceres/ceres.h"
#include <ceres/rotation.h>

using namespace std;
using namespace cv;
using namespace ceres;

#define analyticalBA true
#define autodiffBA false
#define gaussnewtonBA false

double fx = 520.9;
double fy = 521;
double cx = 325.1;
double cy = 249.7;

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
);

struct SnavelyReprojectionError{
    Point3d _p3d;
    Point2d _p2d;
    SnavelyReprojectionError(Point2d p2d, Point3d p3d):_p2d(p2d), _p3d(p3d){}

    template<typename T>
    bool operator()(const T* const R, const T* const t, T* residual)const{
        T p2[2] = {T(_p2d.x), T(_p2d.y)};
        T p3_translated[3];
        T p3[3] = {T(_p3d.x), T(_p3d.y), T(_p3d.z)};
        ceres::AngleAxisRotatePoint(R, p3, p3_translated);
        p3_translated[0] += t[0];
        p3_translated[1] += t[1];
        p3_translated[2] += t[2];

        T reprojected_x = p3_translated[0]*fx/p3_translated[2] + cx;
        T reprojected_y = p3_translated[1]*fy/p3_translated[2] + cy;

        residual[0] = p2[0] - reprojected_x;
        residual[1] = p2[1] - reprojected_y;
        return true;
    }
};

class BA_analytical:public SizedCostFunction<2,6>{
public:
    BA_analytical(Point2d observed_p, Point3d P3d):_observed_p(observed_p), _P3d(P3d) {}
    virtual ~BA_analytical(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians)const{
        //不优化3d点时，将点看作构造函数的参数
        //pose必须用Sophus的李代数传，不然导数的意义何在呢？jacobian和parameter的维度要一致的
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> T_se3(*parameters);
        Sophus::SE3d T_SE3 = Sophus::SE3d::exp(T_se3);
        Eigen::Vector3d Pw, Pc;
        Pw<<_P3d.x, _P3d.y, _P3d.z;
        //cout<<"2d point: "<<_observed_p.x<<", "<<_observed_p.y<<"\t"<<"Pw = "<<Pw.transpose()<<endl<<"\tSE3 = "<<T_SE3.matrix()<<endl;
        //t<<parameters[0][4], parameters[0][5], parameters[0][6];    //这里就溢出了？写成[1][1]这样就会溢出
        //Eigen::Quaterniond q(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        //Eigen::Matrix3d R = q.toRotationMatrix();
        Pc =  T_SE3*Pw;
        //cout<<"Pc = "<<Pc.transpose()<<endl;
        double X = Pc(0);
        double Y = Pc(1);
        double Z = Pc(2);
        double X2 = X*X;
        double Y2 = Y*Y;
        double Z2 = Z*Z;

        residuals[0] = _observed_p.x - cx - fx*X/Z; //residual要和jacobian对应起来,对应起来了
        residuals[1] = _observed_p.y - cy - fy*Y/Z;

        if(!jacobians) return true; //为什么要保证jacobians不为NULL且jacobians[0]不为NULL？
        double* jacobian = jacobians[0];
        if(!jacobian) return true;
        jacobian[3] = fx*X*Y/Z2;
        jacobian[4] = -fx - fx*X2/Z2;
        jacobian[5] = fx*Y/Z;
        jacobian[0] = -fx/Z;
        jacobian[1] = 0;
        jacobian[2] = fx*X/Z2;

        jacobian[9] = fy + fy*Y2/Z2;
        jacobian[10] = -fy*X*Y/Z2;
        jacobian[11] = -fy*X/Z;
        jacobian[6] = 0;
        jacobian[7] = -fy/Z;
        jacobian[8] = fy*Y/Z2;
        return true;
    }
private:
    Point2d _observed_p;
    Point3d _P3d;
};

class BA_all_analytical:public SizedCostFunction<2, 3, 6>{
public:
    BA_all_analytical(double observed_x, double observed_y):_observed_x(observed_x), _observed_y(observed_x) {}
    virtual ~BA_all_analytical(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians)const{
        //1.怎么区分多个block jacobian是按[0],[1]区分，不知parameters是否也如此
        //2.用李代数写雅可比和参数
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> T_se3(parameters[1]); //为啥都要用map来写？
        Sophus::SE3d T_SE3 = Sophus::SE3d::exp(T_se3);
        Eigen::Vector3d Pw, Pc, t;
        Pw<<parameters[0][0], parameters[0][1], parameters[0][2];
        //Eigen::Matrix<double, 3, 4> T = T_SE3.matrix();
        Pc =  T_SE3*Pw;
        double X = Pc(0);
        double Y = Pc(1);
        double Z = Pc(2);
        double X2 = X*X;
        double Y2 = Y*Y;
        double Z2 = Z*Z;

        residuals[0] = _observed_x - cx - fx*X/Z;
        residuals[1] = _observed_y - cy - fy*Y/Z;


        //jacobian of point block
        if(jacobians && jacobians[0]){
            Eigen::Matrix<double, 2, 3> jacobian;
            jacobian(0,0) = -fx/Z;
            jacobian(0,1) = 0;
            jacobian(0,2) = fx*X/Z2;
            jacobian(1,0) = 0;
            jacobian(1,1) = -fy/Z;
            jacobian(1,2) = fy*Y/Z2;
            jacobian = jacobian*T_SE3.rotationMatrix();
            int pos = 0;
            for(int row = 0; row<jacobian.rows(); row++)
                for(int col = 0; col<jacobian.cols(); col++)
                    jacobians[0][pos++] = jacobian(row, col);
        }

        //jacobian of pose(R, t)
        if(jacobians && jacobians[1]){
            jacobians[1][0] = -fx/Z;
            jacobians[1][1] = 0;
            jacobians[1][2] = fx*X/Z2;
            jacobians[1][3] = fx*X*Y/Z2;
            jacobians[1][4] = -fx - fx*X2/Z2;
            jacobians[1][5] = fx*Y/Z;

            jacobians[1][6] = 0;
            jacobians[1][7] = -fy/Z;
            jacobians[1][8] = fy*Y/Z2;
            jacobians[1][9] = fy + fy*Y2/Z2;
            jacobians[1][10] = -fy*X*Y/Z2;
            jacobians[1][11] = -fy*X/Z;
        }

        return true;

        /*if(!jacobians) return true; //为什么要保证jacobians不为NULL且jacobians[0]不为NULL？
        double* jacobian = jacobians[5];
        double* jacobian_p = jacobians[0];
        if(!jacobian || !jacobian_p) return true;
        jacobian_p[0] = ;
        jacobian_p[1] = ;
        jacobian_p[2] = ;

        jacobian_p[3] = ;
        jacobian_p[4] = ;
        jacobian_p[5] = ;


        jacobian[3] = fx*X*Y/Z2;
        jacobian[4] = -fx - fx*X2/Z2;
        jacobian[5] = fx*Y/Z;
        jacobian[0] = -fx/Z;
        jacobian[1] = 0;
        jacobian[2] = fx*X/Z2;

        jacobian[9] = fy + fy*Y2/Z2;
        jacobian[10] = -fy*X*Y/Z2;
        jacobian[11] = -fy*X/Z;
        jacobian[6] = 0;
        jacobian[7] = -fy/Z;
        jacobian[8] = fy*Y/Z2;*/
        return true;
    }
private:
    double _observed_x;
    double _observed_y;
};

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }



    //解析求导
    //opencv mat->旋转矩阵
    Eigen::Matrix3d R_matrix;
    R_matrix<<R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    Eigen::Quaterniond q(R_matrix);
    Eigen::Vector3d t_eigen;
    t_eigen<<t.at<double>(0), t.at<double>(1), t.at<double>(2); //input要用se3输入
    Sophus::SE3d SE3_Rt(R_matrix, t_eigen);
    Sophus::Vector6d se3 = SE3_Rt.log();   //se3要用PNP的结果做初值
    //Sophus::Vector6d se3;   //也可用别的值来看初值对优化结果及时间的影响
    //double r_esti_autodiff[3] = {r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0)};    //R按旋转向量给,t按平移向量给
    //double t_esti_autodiff[3] = {t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
    double r_esti_autodiff[3] = {};    //R按旋转向量给,t按平移向量给
    double t_esti_autodiff[3] = {};     //也可用别的值来看初值对优化的影响
    cout<<"calling bundle adjustment by ceres using analytical derivative"<<endl;
    Problem problem;
    if(analyticalBA){
        for(int i = 0; i<pts_3d.size(); i++){
            double p3d[3] = {pts_3d[i].x, pts_3d[i].y, pts_3d[i].z};
            CostFunction* costfunction = new BA_all_analytical(pts_2d[i].x, pts_2d[i].y);
            problem.AddResidualBlock(costfunction, NULL, p3d, se3.data());
            /*
            //解析求导不考虑点
            CostFunction* costfunction = new BA_analytical(pts_2d[i], pts_3d[i]);
            problem.AddResidualBlock(costfunction, NULL, se3.data());*/

        }
        cout<<"analytical BA begin..."<<endl;
    }
    if(autodiffBA){
        for(int i = 0; i<pts_3d.size(); i++){
            // 自动求导方法
            CostFunction* costfunction_autodiff = new AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3>(
                    new SnavelyReprojectionError(pts_2d[i], pts_3d[i])
            );
            problem.AddResidualBlock(costfunction_autodiff, NULL, r_esti_autodiff, t_esti_autodiff);
        }
        cout<<"Autodiff BA begin..."<<endl;
    }
        Solver::Options options;
        options.function_tolerance = 1e-20;
        options.parameter_tolerance = 1e-20;
        //options.initial_trust_region_radius = 1e-40;
        options.min_trust_region_radius = 1e-40;
        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        chrono::steady_clock::time_point t_begin = chrono::steady_clock::now();
        Solve(options,&problem, &summary);
        chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
        chrono::duration<double> cost_time = chrono::duration_cast<chrono::duration<double>>(t_end-t_begin);
        cout<<"time cost in my Analytical BA is:"<<cost_time.count()<<endl;
        cout<<summary.BriefReport()<<endl;   //这里发生了seg default段错误？？？
        cout<<"estimation result: "<<se3.transpose()<<endl;

    /*if(gaussnewtonBA){
        cout << "calling bundle adjustment by gauss newton" << endl;
        Sophus::SE3d pose_gn;
        t1 = chrono::steady_clock::now();
        bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;
    }*/

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            //cout<<"2d point: "<<points_2d[i].transpose()<<"\t"<<points_3d[i].transpose()<<endl;
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n" << pose.matrix() << endl;
}
