#!/bin/bash
set -e

DATA_FOLDER_RAW=./data/raw
DATA_FOLDER=./data
mkdir -p $DATA_FOLDER_RAW
mkdir -p $DATA_FOLDER

gdown https://drive.google.com/uc?id=1U-Hly00DmqtAIGaPkF-Eu_B_q0Frsbh1 -O $DATA_FOLDER_RAW/lf.zip
unzip $DATA_FOLDER_RAW/lf.zip -d $DATA_FOLDER_RAW
rm $DATA_FOLDER_RAW/lf.zip
if [ ! -d "$DATA_FOLDER_RAW/LF" ]; then
    echo "Directory not found: $DATA_FOLDER_RAW/LF"
    exit 1
fi

scenes=(africa basket statue torch)
for scene in ${scenes[@]}; do (
    scene_folder_raw=$DATA_FOLDER_RAW/LF/$scene
    scene_folder=$DATA_FOLDER/lf/$scene
    mkdir -p $scene_folder
    cp -r $scene_folder_raw/colmap $scene_folder
    cp -r $scene_folder_raw/images $scene_folder
    cp -r $scene_folder_raw/images_2 $scene_folder
    cp $scene_folder_raw/transforms.json $scene_folder
) &
done
wait
