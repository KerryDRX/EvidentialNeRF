#!/bin/bash
set -e

DATA_FOLDER_RAW=./data/raw
DATA_FOLDER=./data
mkdir -p $DATA_FOLDER_RAW
mkdir -p $DATA_FOLDER

wget https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz -O $DATA_FOLDER_RAW/robustnerf.tar.gz
tar -xvf $DATA_FOLDER_RAW/robustnerf.tar.gz -C $DATA_FOLDER_RAW
rm $DATA_FOLDER_RAW/robustnerf.tar.gz
if [ ! -d "$DATA_FOLDER_RAW/robustnerf" ]; then
    echo "Directory not found: $DATA_FOLDER_RAW/robustnerf"
    exit 1
fi

scenes=(android crab2 statue yoda)
for scene in ${scenes[@]}; do (
    scene_folder_raw=$DATA_FOLDER_RAW/robustnerf/$scene
    scene_folder=$DATA_FOLDER/robustnerf/$scene
    mkdir -p $scene_folder/colmap
    cp -r $scene_folder_raw/sparse $scene_folder/colmap
    cp $scene_folder_raw/database.db $scene_folder/colmap
    cp -r $scene_folder_raw/images $scene_folder
    cp -r $scene_folder_raw/images_8 $scene_folder
) &
done
wait
