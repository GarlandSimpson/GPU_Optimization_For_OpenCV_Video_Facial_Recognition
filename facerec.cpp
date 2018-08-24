// facerec.cpp : Defines the entry point for the console application.
// TEAM: YouTube Facial Recognition 
// NOTE: Code must be compiled as a win32 app or else it EXPLODES!
//
//

#include "stdafx.h"


/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));  // 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 4) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    // Get the path to your CSV:
    string fn_haar = string(argv[1]);
    string fn_csv = string(argv[2]);
	string fn_vid = string(argv[3]); // int deviceId = atoi(argv[3]);
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
	//vector<gpu::GpuMat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {

        read_csv(fn_csv, images, labels);
		

    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
	cout << "Create model" << endl;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	cout << "Training model" << endl;
    model->train(images, labels);
	cout << "Done Training" << endl;
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
	//gpu::CascadeClassifier_GPU haar_cascade;



    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
	VideoCapture cap(fn_vid);   // VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        // cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		cerr << "You suck" << endl;
        return -1;
    }
	if (cap.grab())
		cout << cap.grab() << endl;
    // Holds the current frame from the Video device:
	double frnb(cap.get(CV_CAP_PROP_FRAME_COUNT));
	double fIdx = 0.0;
    Mat frame;
    for(;;) {


		fIdx = fIdx + 25;
		
		//std::cout << "frame index ? ";
		//std::cin >> fIdx;
		if (fIdx < 0 || fIdx >= frnb) break;
			cap.set(CV_CAP_PROP_POS_FRAMES, fIdx);
			bool success = cap.read(frame);
			if (!success) {
				cout << "Cannot read  frame " << endl;
				break;
			}


		cap >> frame;
		//frame = imread("C:/Users/simps/Desktop/tester_mc_testyfaces/Angela_Bassett/4/aligned_detect_4.1047.jpg");
        // Clone the current frame:
        Mat original = frame.clone();
		//gpu::GpuMat original = frame.clone();

        // Convert the current frame to grayscale:
		//gpu::GpuMat gray;

		//-----------Section that breaks with gpu---------------

        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
		//------------------------------------------------------

        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?

        for(int i = 0; i < faces.size(); i++) {
			
			//-----------Section that breaks with gpu---------------
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
			//gpu::GpuMat face = gray(face_i);
			//------------------------------------------------------

            Mat face_resized;
			//gpu::GpuMat face_resized;

            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:


            // int prediction = model->predict(face_resized);  // <-----------------------------
			int prediction = -1;
			double prediction_confidence = 0.0;

			model->predict(face_resized, prediction, prediction_confidence);
			
			// prediction_confidence = prediction_confidence / 100;

			cout << "prediction confidence: " << prediction_confidence << endl;



            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            string box_text = format("Prediction = %d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			
        }
		
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
