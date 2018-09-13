# inference-tutorials-generic

## [Face Detection Tutorial](face_detection_tutorial/Readme.md)
- Inference Engine Overview
- Key Concepts
  + Floating Point Precision
  + Batch Size (brief)
- Tutorial Steps
  + Tutorial Step 1: Create the Base OpenCV Application
    + gflags library to parse command line arguments
    + Using OpenCV for video input and output to window
  + Tutorial Step 2: Add the first Model, Face Detection
    + Helper functions matU8ToBlob() and Load()
    + BaseDetection class
    + FaceDetectionClass class
  + Tutorial Step 3: Add the Second Model, Age and Gender
    + AgeGenderDetection class
  + Tutorial Step 4: Add the Third Model, Head Pose 
    + HeadPoseDetection class
    + Picking the Right Models for the Right Devices 

## [Car Detection Tutorial](car_detection_tutorial/Readme.md)
- Inference Engine Overview
- Key Concepts
  + Batch Size (detailed)
  + Image Processing Pipeline
  + Synchronous vs. Asynchronous API
- Tutorial Steps
  + Tutorial Step 1: Create the Base OpenCV Application
    + gflags library to parse command line arguments
    + Using OpenCV for video input and output to window
  + Tutorial Step 2: Add the first Model, Vehicle Detection
    + Helper functions matU8ToBlob() and Load()
    + BaseDetection class
    + VehicleDetection class
  + Tutorial Step 3: Add the Second Model, Vehicle Attributes Detection
    + VehicleAttribsDetection class 
    + Running combinations of models on devices (exercise)
  + Tutorial Step 4: Using the Asynchronous API
    + Changes made to Tutorial Step 3 for converting from synchronouse to asynchronous
    + Synchronous vs. Asynchronous performance (exercise)
