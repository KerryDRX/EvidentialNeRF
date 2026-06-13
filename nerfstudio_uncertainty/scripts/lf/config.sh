datadir=data
outputdir=output
dataset=lf
scenes=(africa basket statue torch)
steps_per_eval=1000
n_cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)

get_scene_schedule() {
    local scene=$1
    case $scene in
        africa)
            scene_steps=5000
            scene_warmup=1000
            ;;
        basket)
            scene_steps=10000
            scene_warmup=4000
            ;;
        statue)
            scene_steps=10000
            scene_warmup=7000
            ;;
        torch)
            scene_steps=5000
            scene_warmup=1000
            ;;
        *)
            echo "Unknown scene: $scene" >&2
            return 1
            ;;
    esac
}
