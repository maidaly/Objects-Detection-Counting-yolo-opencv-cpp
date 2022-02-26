#include "include/yolo.h"
#include <filesystem>


vector<Mat> read_images(String dir) {
    vector<String> filenames;
    vector<Mat> images;
    cv::glob(dir + "/*.jpg", filenames);

    for (auto file : filenames) {
        Mat img = imread(file);
        images.push_back(img);
    }
    return images;
}

void detectAndWriteVideo(String input_dir, String output_dir, YOLO yolo_model){
    for (const auto &entry :  std::filesystem::directory_iterator(input_dir)) {
        String view = entry.path().string();
        vector<Mat> images = read_images(view);
        VideoWriter video(output_dir + view.substr(view.size() - 8, view.size()) + ".avi",
                          cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                          7, Size(768, 576));

        for (auto img : images) {
            Mat srcimg = img;
            for (auto roi:rois) {
                rectangle(img,
                          Point(roi.left, roi.top),
                          Point(roi.right, roi.bottom), Scalar(0, 0, 255),
                          1.5);
            }
            yolo_model.detect(srcimg);
            video.write(srcimg);

        }
        video.release();
    }
}


int main( ) {

//    String time13_57_dir = "D:\\opencv_object_c++\\data\\S1_L1\\Crowd_PETS09\\S1\\L1\\Time_13-57\\";
    String time13_59_dir = "D:\\opencv_object_c++\\data\\S1_L1\\Crowd_PETS09\\S1\\L1\\Time_13-59\\";
    string out1_dir = "../output/Time_13-57/";
    string out2_dir ="../output/Time_13-59/";
    Mat srcimg = read_images("D:\\opencv_object_c++\\data\\S1_L1\\Crowd_PETS09\\S1\\L1\\Time_13-57\\View_001")[0];
    YOLO yolo_model(yolo_nets);
    detectAndWriteVideo(time13_57_dir, out1_dir, yolo_model);
    detectAndWriteVideo(time13_59_dir, out2_dir, yolo_model);

}

