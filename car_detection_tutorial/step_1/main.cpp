/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_Vehicle_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>
#include "car_detection.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    return true;
}


int main(int argc, char *argv[]) {
    try {
        /** This sample covers 3 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ---------------------------Parsing and validation of input args--------------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // -----------------------------Read input -----------------------------------------------------
        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        /** requesting new frame if any*/
        cap.grab();

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto wallclock = std::chrono::high_resolution_clock::now();

        double ocv_render_time = 0;
        bool firstFrame = true;
        bool doMoreFrames = true;
        /** Start inference & calc performance **/
        do {
        	std::chrono::high_resolution_clock::time_point t0,t1;

            if (firstFrame && !FLAGS_no_show) {
                slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
            }

            firstFrame = false;

			// -----------------------Display Results ---------------------------------------------
			t0 = std::chrono::high_resolution_clock::now();
			if (!FLAGS_no_show) {
				cv::imshow("Detection results", frame);
//				cv::waitKey(0);
			}
			t1 = std::chrono::high_resolution_clock::now();
			ocv_render_time += std::chrono::duration_cast<ms>(t1 - t0).count();

			// get next frame            
           	doMoreFrames = cap.read(frame);

			// watch for keypress to stop or snapshot
			int keyPressed;
			if (-1 != (keyPressed = cv::waitKey(1)))
			{
				if ('s' == keyPressed) {
					// save screen to output file
					slog::info << "Saving snapshot of image" << slog::endl;
					cv::imwrite("snapshot.bmp", frame);
				} else {
					doMoreFrames = false;
				}
			}

            // end of file we just keep last image/frame displayed to let user check what was shown
            if (!doMoreFrames && !FLAGS_no_wait && !FLAGS_no_show) {
                slog::info << "Press 's' key to save a snapshot, press any other key to exit" << slog::endl;
                while (cv::waitKey(0) == 's') {
            		// save screen to output file
            		slog::info << "Saving snapshot of image" << slog::endl;
            		cv::imwrite("snapshot.bmp", frame);
                }
                doMoreFrames = false;
                break;
            }

        } while(doMoreFrames);
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
