# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# use relative location of script to specify where to find downloaded models
scriptDir=`cd $(dirname $BASH_SOURCE); pwd`
modelDir=`cd $scriptDir/../../../tutorial_models/face_detection; pwd`

# Face Detection Model - ADAS
modName=face-detection-adas-0001
export mFDA16=`find $modelDir -name "${modName}-fp16.xml"`
export mFDA32=`find $modelDir -name "${modName}.xml"`

# Face Detection Model - Retail
modName=face-detection-retail-0004
export mFDR16=`find $modelDir -name "${modName}-fp16.xml"`
export mFDR32=`find $modelDir -name "${modName}.xml"`

# Age and Gender Model
modName=age-gender-recognition-retail-0013
export mAG16=`find $modelDir -name "${modName}-fp16.xml"`
export mAG32=`find $modelDir -name "${modName}.xml"`

# Head Pose Estimation Model
modName=head-pose-estimation-adas-0001
export mHP16=`find $modelDir -name "${modName}-fp16.xml"`
export mHP32=`find $modelDir -name "${modName}.xml"`

