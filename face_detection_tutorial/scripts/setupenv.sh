# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?"Must source the setupvars.sh in the SDK to set InferenceEngine_DIR"}

modelDir=$InferenceEngine_DIR/../../intel_models

# Face Detection Model - ADAS
modName=face-detection-adas-0001
export mFDA16=$modelDir/$modName/FP16/$modName.xml
export mFDA32=$modelDir/$modName/FP32/$modName.xml

# Face Detection Model - Retail
modName=face-detection-retail-0004
export mFDR16=$modelDir/$modName/FP16/$modName.xml
export mFDR32=$modelDir/$modName/FP32/$modName.xml

# Age and Gender Model
modName=age-gender-recognition-retail-0013
export mAG16=$modelDir/$modName/FP16/$modName.xml
export mAG32=$modelDir/$modName/FP32/$modName.xml

# Head Pose Estimation Model
modName=head-pose-estimation-adas-0001
export mHP16=$modelDir/$modName/FP16/$modName.xml
export mHP32=$modelDir/$modName/FP32/$modName.xml

