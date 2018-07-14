# Tutorial Step 1: Create the Base OpenCV Application

![image alt text](../doc_support/step1_image_0.png)

# Table of Contents

<p></p><div class="table-of-contents"><ul><li><a href="#tutorial-step-1-create-the-base-opencv-application">Tutorial Step 1: Create the Base OpenCV Application</a></li><li><a href="#table-of-contents">Table of Contents</a></li><li><a href="#introduction">Introduction</a></li><li><a href="#the-basic-opencv-application-input-and-output">The Basic OpenCV Application, Input and Output</a><ul><li><a href="#parsing-command-line-arguments">Parsing Command Line Arguments</a><ul><li><a href="#create-the-argument">Create the Argument</a></li><li><a href="#parse-arguments">Parse Arguments</a></li></ul></li><li><a href="#opencv-input-to-output">OpenCV Input to Output</a><ul><li><a href="#header-files">Header Files</a></li><li><a href="#main">main()</a></li><li><a href="#main-loop">Main Loop</a></li></ul></li></ul></li><li><a href="#building-and-running">Building and Running</a><ul><li><a href="#build">Build</a></li><li><a href="#run">Run</a></li></ul></li><li><a href="#conclusion">Conclusion</a></li><li><a href="#navigation">Navigation</a></li></ul></div><p></p>

# Introduction

This tutorial will show the basics of what is needed to include and use OpenCV in an application. We will be walking through the sample application that has already been created.  The sample is designed to be a minimal application that demonstrates how to use OpenCV functions to read image data and then display the image data.  This tutorial will walkthrough the OpenCV portions of the code and explain what it does.  Then we will build and run the tutorial so we can see it in action.  In later tutorials, we will be adding processing of the input image to this basic framework.

# The Basic OpenCV Application, Input and Output

Every application needs some way of getting data in and data out.  Let us now take a look at the code we will be using to do the input and output in our OpenCV application.  Then we can compile and run our program to see how it works using the base input and output settings.  

## Parsing Command Line Arguments

To make it easier to set everything from the input video file to which model and device is to be used, command line arguments to the application will be used.  To parse the command line arguments, the application will use the "gflags" helper library that comes with the OpenVINO™ toolkit samples.  Here we will briefly go over the primary functions that are used, for reference the full source code for the gflags library may be found in the OpenVINO™ toolkit samples directory: 

```bash
/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/thirdparty/gflags
```


To make use of the gflags library and use the supplied functions and classes, the main header file must be included:

```cpp
#include <gflags/gflags.h>
```


This is done in the main header file "face_detection.hpp" where all the command line arguments are defined using the following steps.

### Create the Argument

Create the argument using the macro "DEFINE_string".  Here the “-i \<video filename\>” argument that will be used to specify the input video is defined:

```cpp
/// @brief message for images argument
static const char video_message[] = "Optional. Path to an video file. Default value is \"cam\" to work with camera.";

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);
```


In the above code:

* video_message[] is the argument’s help string

* DEFINE_string(i, "cam", video_message):

   * Specifies:

      * The string argument name as "i"

      * video_message as the help message

      * "cam" as the default value when not set

   * Creates the variable FLAGS_i to hold the string value for the "i" argument

### Parse Arguments

In main.cpp’s main() function, ParseAndCheckCommandLine() is called to do command line argument parsing and checking for valid arguments. The actual argument parsing and setting variables is done by the call:

```cpp
gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
```


After returning, the FLAGS_i variable will be set with the "-i" command line arguments “<video filename>” value, or if not specified, the default value of “cam”.

This is how the "-i" argument is done, all other arguments are handled similarly using the other forms of the gflags macro according to data type needed as follows:

* DEFINE_uint32() for an unsigned 32-bit integer arguments

* DEFINE_string() for string arguments

* DEFINE_double() for double precision floating point arguments

* DEFINE_bool() for boolean arguments

## OpenCV Input to Output

1. Open up a terminal (such as xterm) or use an existing terminal to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 1:

```bash
cd tutorials/face_detection_tutorial/
cd step_1
```


3. Open the files "main.cpp" and “face_detection.hpp” in the editor of your choice such as ‘gedit’, ‘gvim’, or ‘vim’.

### Header Files

1. These header files are included to define helpful utility classes used to simplify common tasks as well as some functions for making logging easier.

```cpp
#include <samples/common.hpp>
#include <samples/slog.hpp>
```


2. The opencv.hpp file is included for the Intel® optimized OpenCV libraries included in the OpenVINO™ toolkit. 

```cpp
#include <opencv2/opencv.hpp>
```


### main()

1. First, the OpenCV video capture object "cap" is created that will be used to source the image data.  Then, the image source is opened.  FLAGS_i is the command line parameter that tells the application the source of where the image.  The source can be the path to an image file, the path to a video file, or "cam" for the USB camera.

```cpp
cv::VideoCapture cap;
if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
   throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
}
```


2. The width and height of the image source are stored for use later.  

```cpp
const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
```


3. Storage for the image frame is created and then the first frame is read in. 

```cpp
cv::Mat frame;
if (!cap.read(frame)) {
   throw std::logic_error("Failed to get frame from cv::VideoCapture");
}
```


### Main Loop

The main loop will read in and then write out the image frames until there are no more available. 

1. The main loop runs forever until a break elsewhere inside the loop when there are no more images:

```cpp
while (true) {
```


2. An available image from the source is grabbed:

```cpp
    /** requesting new frame if any*/
    cap.grab();
```


3. A check is done for a key press to either saving a snapshot (pressing ‘s’) or stopping (any other key) the application.

```cpp
    int keyPressed;
    if (-1 != (keyPressed = cv::waitKey(1)))
    {
   	 if ('s' == keyPressed) {
   		 // save screen to output file
   		 slog::info << "Saving snapshot of image" << slog::endl;
   		 cv::imwrite("snapshot.bmp", frame);
   	 } else {
   		 break;
   	 }
    }
```


4. The output is shown, wrapped with time functions to measure the time it took to do:

```cpp
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!FLAGS_no_show) {
   	  cv::imshow("Detection results", frame);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
```


5. A check is made to see if there is another image to process.  If there is nothing to retrieve then a key press is waited for if the command line parameter "-no_wait" was not used.

```cpp
    // end of file, for single frame file, like image we just keep it displayed to let user check what was shown
    cv::Mat newFrame;
    if (!cap.retrieve(newFrame)) {
   	  if (!FLAGS_no_wait) {
   		   slog::info << "Press 's' key to save a snapshot, press any other key to exit" << slog::endl;
   		   while (cv::waitKey(0) == 's') {
   			 // save screen to output file
   			 slog::info << "Saving snapshot of image" << slog::endl;
   			 cv::imwrite("snapshot.bmp", frame);
   		   }
   	  }
   	  break;
    }
    frame = newFrame;  // shallow copy
```


6. A message is output to let the user know they can stop a multi-image source like video or camera.

```cpp
    if (firstFrame) {
   	  slog::info << "Press 's' key to save a snapshot, press any other key to stop" << slog::endl;
    }

    firstFrame = false;
```


7. The loop ends and goes back to the while(true) at the top to run again

```cpp
}
```


# Building and Running

Now that we have looked at the code and understand how the program works, let us compile and run to see it in action.

## Build

1. First, we need to configure the build environment when using the OpenVINO™ toolkit by running the "setupvars.sh" script.

```bash
source  /opt/intel/computer_vision_sdk/bin/setupvars.sh
```


2. Now we need to create a directory to build the tutorial in and change to it.

```bash
mkdir build
cd build
```


3. The last thing we need to do before compiling is to configure the build settings and build the executable.  We do this by running CMake to setup the build target and file locations.  Then, we run Make to build the executable:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```


4. You should now have the executable "face_detection_tutorial" file in the “./intel64/Release/” directory.  We will be using that executable to run our base OpenCV application.

## Run

1. Now, it is time to run the application.  We will run it using each type of input (image file, video file, camera) so you will know what to expect.  We have included commands that will have the application load images or videos that come with the OpenVINO™ toolkit and this tutorial, but you can also use your own images.  If the application cannot find the specified input image file or if a USB camera cannot be found, it will print an error message and return to the command prompt.  If that happens, check the path to the image or video file, to make sure it is correct and try again.

2. First, let us use the application to view a single image file.  We do this by using a "-i" parameter followed by the name of an image file.

```bash
./intel64/Release/face_detection_tutorial -i ../../data/face.jpg
```


3. You should now see a new window with an image.  You should also see a "Press 's' key to save a snapshot, press any other key to exit" prompt in the console window.  The application will now wait for you to press a key with the image window active.

**Note**: Pressing a key in the console window will not do anything because the image window is detecting key presses.  Use Ctrl+C to exit.

4. Next, let us see how the application handles a video file:

```bash
./intel64/Release/face_detection_tutorial -i /opt/intel/computer_vision_sdk/openvx/samples/samples/face_detection/face.mp4
```


5. You will now see a window appear and play the video.  After the video has finished playing, the window will continue to display the final frame of the video, waiting for you to press a key with the image window active.

6. [Optional]Finally, if a USB camera has been setup, we can use the application to view live video from the connected USB camera.  The camera is the default source, so we do this by running the application without using any parameters.

```bash
./intel64/Release/face_detection_tutorial
```


Or we can still specify the camera using "cam":

```bash
./intel64/Release/face_detection_tutorial -i cam
```


7. [Optional]You will now see the output window appear displaying live input from the USB camera.  When you are ready to exit the application, make sure the output window is active and press a key.

# Conclusion

Now we have seen what it takes to create a basic application that uses OpenCV to read and display image data.  We have also seen how the application works with each type of image input it accepts including still images, video files, and live video from an optional USB camera.  We will be using the basic framework from this step of the tutorial as we move forward building up the application step-by-step.  Next, in Tutorial Step 2, we will be adding the ability to process images and actually detect faces.

# Navigation

[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 2](../step_2/Readme.md)

