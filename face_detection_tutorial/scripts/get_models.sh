# This script uses the model downloader script located in the OpenVINO installation
# to download all the necessary models used in the tutorial

# model downloader Python script
modelDownloader=/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py

# use relative location of script to specify where to put downloaded models
scriptDir=`cd $(dirname $BASH_SOURCE); pwd`
mkdir $scriptDir/../../../tutorial_models
mkdir $scriptDir/../../../tutorial_models/face_detection
modelDir=`cd $scriptDir/../../../tutorial_models/face_detection; pwd`

# prefix to see output from this script 
prefix="[get_models.sh]:"

echo "${prefix}Models will be downloaded to: $modelDir"

modName=face-detection-adas-0001*
# will also get face-detection-adas-0001-fp16
echo "${prefix}Downloading model $modName"
$modelDownloader --name $modName --output_dir $modelDir

modName=face-detection-retail-0004*
# will also get face-detection-retail-0004-fp16
echo "${prefix}Downloading model $modName"
$modelDownloader --name $modName --output_dir $modelDir

modName=age-gender-recognition-retail-0013*
# will also get age-gender-recognition-retail-0013-fp16
echo "${prefix}Downloading model $modName"
$modelDownloader --name $modName --output_dir $modelDir

modName=head-pose-estimation-adas-0001*
# will also get head-pose-estimation-adas-0001-fp16
echo "${prefix}Downloading model $modName"
$modelDownloader --name $modName --output_dir $modelDir

echo "${prefix}: Models have been downloaded to: $modelDir"
for f in `find $modelDir -name "*.xml"`; do
	echo "${prefix}Downloaded model: $f"
done

