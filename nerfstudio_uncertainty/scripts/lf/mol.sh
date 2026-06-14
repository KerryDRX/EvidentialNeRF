#!/bin/bash
set -Eeuo pipefail

cleanup() {
    trap - SIGINT SIGTERM
    kill 0
    exit 130
}
trap cleanup SIGINT SIGTERM
source nerfstudio_uncertainty/scripts/lf/config.sh

method=nerfacto-mol

run() {
    local scene=$1
    local gpu=${2:-0}
    local seed=${3:-0}
    export CUDA_VISIBLE_DEVICES=$gpu

    local steps
    local warmup
    get_scene_schedule $scene
    steps=$scene_steps
    warmup=$scene_warmup

    cores_per_job=$((n_cores / n_jobs))
    start_core=$((gpu * cores_per_job))
    end_core=$((start_core + cores_per_job - 1))
    experiment=$dataset/$scene/seed$seed
    eval_outputdir=$outputdir/$experiment/$method/main

    taskset -c $start_core-$end_core ns-train $method \
        --output-dir $outputdir --experiment-name $experiment --data $datadir/$dataset/$scene \
        --steps-per-save $steps_per_eval --steps-per-eval-all-images $steps_per_eval --max-num-iterations $(($steps+1)) \
        --save-only-latest-checkpoint True \
        \
        --machine.seed $seed \
        \
        --pipeline.model.near-plane 1.0 --pipeline.model.far-plane 100.0 \
        --pipeline.model.background-color last_sample \
        --pipeline.model.max-res 4096 \
        --pipeline.model.proposal-initial-sampler uniform \
        --pipeline.model.distortion-loss-mult 0.0 \
        --pipeline.model.disable-scene-contraction True \
        \
        --pipeline.model.warmup_steps $warmup \
        \
        $dataset --scene $scene --downscale-factor 2 --auto-scale-poses False

    taskset -c $start_core-$end_core ns-eval \
        --load-config $eval_outputdir/config.yml \
        --output-path $eval_outputdir/output.json \
        --render-output-path $eval_outputdir/outputs
}

n_jobs=4
for i in {0..3}; do (
    scene=${scenes[$i]}
    echo "Running $scene"
    for seed in {0..14}; do
        run $scene $i $seed
    done
) &
done
wait
