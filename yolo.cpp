//
// Created by MaiDaly on 2/16/2022.
//

#include "include/yolo.h"


int roi_contain_object(Coord roi_coord, int left, int top, int right, int bottom) {
    int box_hight = bottom - top;
    if ((roi_coord.right > right && right > roi_coord.left
         && roi_coord.bottom > bottom
         && roi_coord.bottom - top > box_hight / 2) ||
        (roi_coord.right > left
         && left > roi_coord.left
         && roi_coord.bottom > top
         && roi_coord.bottom - top > box_hight / 2)) {
//        cout<<"object detected on roi";
        return 1;
    } else { return 0; }
}

void annotateRoi(Mat &frame, vector<Coord> rois, vector<int> rois_count) {
    for (size_t i = 0; i < rois_count.size(); i++) {
        Coord local_roi = rois[i];
//        cout<<local_roi.bottom<<" "<<local_roi.left<<endl;
        string roi_label = format("ROI:%i", i + 1);
//        cout << rois_count[i] << endl;
        string count_label = format("People count:%i", rois_count[i]);
        int baseLine = 0;
        Size roi_labelSize = getTextSize(roi_label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Size count_labelSize = getTextSize(roi_label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        putText(frame, roi_label,
                Point(local_roi.left+5, local_roi.bottom-(count_labelSize.height+roi_labelSize.height+5)),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1);
        putText(frame, count_label,
                Point(local_roi.left+5, local_roi.bottom- count_labelSize.height),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1);
    }
}

YOLO::YOLO(Net_config config) {
    cout << "Loading the model: " << config.netname << endl;
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->inpWidth = config.inpWidth;
    this->inpHeight = config.inpHeight;
    strcpy(this->netname, config.netname.c_str());

    ifstream ifs(config.classesFile.c_str());
    string line;
    while (getline(ifs, line)) this->classes.push_back(line);

    this->net = readNetFromDarknet(config.modelConfiguration, config.modelWeights);
    this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(DNN_TARGET_CPU);
}

void YOLO::drawBox(Mat &img, int left, int top, int right, int bottom, Scalar color) {
    int line_width = (right - left) /4 ;
    rectangle(img,
              Point(left, top),
              Point(right, bottom),
              color,
              1);
    line(img, Point(left, top), Point(left + line_width, top), color, 2);
    line(img, Point(left, top), Point(left, top + line_width), color, 2);
    line(img, Point(right, bottom), Point(right - line_width, bottom), color, 2);
    line(img, Point(right, bottom), Point(right, bottom - line_width), color, 2);
    line(img, Point(left, bottom), Point(left, bottom - line_width), color, 2);
    line(img, Point(left, bottom), Point(left + line_width, bottom), color, 2);
line(img, Point(right, top), Point(right, top + line_width), color, 2);
line(img, Point(right, top), Point(right - line_width, top), color, 2);

}

void YOLO::postprocess(Mat &frame, const vector<Mat> &outs, vector<Coord> rois) {

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    int num_rois = rois.size();
    vector<int> rois_count(num_rois, 0);     //roi vector array with length of number of rois
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > this->confThreshold) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        if (classIds[idx] == 0) {
            for (size_t i = 0; i < num_rois; i++) {
                Coord roi = rois[i];
                if (roi_contain_object(roi, box.x, box.y, box.x + box.width, box.y + box.height)) {
                    rois_count[i] += 1;
                }

            }

            this->drawBox(frame, box.x, box.y, box.x + box.width, box.y + box.height, Scalar (0, 200, 255));

        }

    }
    annotateRoi(frame, rois, rois_count);
}


void YOLO::detect(Mat &frame) {
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
//    std::cout<<"input set";
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    this->postprocess(frame, outs, rois);
//    std::cout<<"data postprocess done";
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
//    string label = format("%s Inference time : %.2f ms", this->netname, t);
//    putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
//    std::cout<<"detection done";

}



