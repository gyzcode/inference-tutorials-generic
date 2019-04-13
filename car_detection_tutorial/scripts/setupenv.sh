# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# use relative location of script to specify where to find downloaded models
scriptDir=`cd $(dirname $BASH_SOURCE); pwd`
modelDir=`cd $scriptDir/../../../tutorial_models; pwd`

# Vehicle and License Plates Detection Model
modName=vehicle-license-plate-detection-barrier-0106
export mVLP16=`find $modelDir -name "${modName}-fp16.xml"`
export mVLP32=`find $modelDir -name "${modName}.xml"`

# Vehicle-only Detection Model used with the batch size exercise
modName=vehicle-detection-adas-0002
export mVDR16=`find $modelDir -name "${modName}-fp16.xml"`
export mVDR32=`find $modelDir -name "${modName}.xml"`

# Vehicle Attributes Detection Model
modName=vehicle-attributes-recognition-barrier-0039
export mVA16=`find $modelDir -name "${modName}-fp16.xml"`
export mVA32=`find $modelDir -name "${modName}.xml"`


