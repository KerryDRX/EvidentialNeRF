#!/bin/bash
set -e

DATA_FOLDER_RAW=./data/raw
DATA_FOLDER=./data
mkdir -p $DATA_FOLDER_RAW
mkdir -p $DATA_FOLDER

gdown https://drive.google.com/uc?id=11PhkBXZZNYTD2emdG1awALlhCnkq7aN- -O $DATA_FOLDER_RAW/llff.zip
unzip $DATA_FOLDER_RAW/llff.zip -d $DATA_FOLDER_RAW
rm $DATA_FOLDER_RAW/llff.zip
if [ ! -d "$DATA_FOLDER_RAW/nerf_llff_data" ]; then
    echo "Directory not found: $DATA_FOLDER_RAW/nerf_llff_data"
    exit 1
fi

scenes=(fern flower fortress horns leaves orchids room trex)
for scene in ${scenes[@]}; do (
    scene_folder_raw=$DATA_FOLDER_RAW/nerf_llff_data/$scene
    scene_folder=$DATA_FOLDER/llff/$scene
    mkdir -p $scene_folder/colmap
    cp -r $scene_folder_raw/sparse $scene_folder/colmap
    cp $scene_folder_raw/database.db $scene_folder/colmap
    cp -r $scene_folder_raw/images $scene_folder
) &
done
wait
