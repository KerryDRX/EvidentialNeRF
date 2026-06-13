#!/bin/bash
set -e

DATA_FOLDER_RAW=./data/raw
DATA_FOLDER=./data
mkdir -p $DATA_FOLDER_RAW/phototourism
mkdir -p $DATA_FOLDER/phototourism

scenes=(
    brandenburg_gate
    buckingham_palace
    colosseum_exterior
    grand_place_brussels
    notre_dame_front_facade
    palace_of_westminster
    pantheon_exterior
    taj_mahal
    temple_nara_japan
    trevi_fountain
)

for scene in ${scenes[@]}; do (
    wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/$scene.tar.gz -O $DATA_FOLDER_RAW/phototourism/$scene.tar.gz
    tar -xvzf $DATA_FOLDER_RAW/phototourism/$scene.tar.gz -C $DATA_FOLDER/phototourism
) &
done
wait
