//
// The core of the code is: https://github.com/hpc203/yolov34-cpp-opencv-dnn
// I made some modifications to suite my case
//

//#ifndef OPENCV_OBJECT_C___YOLO_H
//
//#define OPENCV_OBJECT_C___YOLO_H
//
//#endif //OPENCV_OBJECT_C___YOLO_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config {
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    int inpWidth;  // Width of network's input image
    int inpHeight; // Height of network's input image
    string classesFile;
    string modelConfiguration;
    string modelWeights;
    string netname;
};

struct Coord {
    int left;
    int top;
    int right;
    int bottom;
};

class YOLO {
public:
    YOLO(Net_config config);

    void detect(Mat &frame);

    void drawBox(Mat &img, int left, int top, int right, int bottom, Scalar color);

private:
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    char netname[20];
    vector<string> classes;
    Net net;

    void postprocess(Mat &frame, const vector<Mat> &outs, vector<Coord> rois);
};

static Net_config yolo_nets = {0.5, 0.4, 416, 416, "coco.names", "../pretrained_models/enet-coco.cfg",
                               "../pretrained_models/enetb0-coco_final.weights", "e-net"};
static vector<Coord> rois = {{10,  8,   757, 566},
                             {287, 156, 711, 431},
                             {27,  129, 230, 289}};