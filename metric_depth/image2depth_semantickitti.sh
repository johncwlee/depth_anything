CKPT_PATH="/home/johnl/misc/depth_anything/depth_anything_metric_depth_outdoor.pt"
set -e
exeFunc(){
    CUDA_VISIBLE_DEVICES=0 python3 predict.py \
    -m zoedepth -d semantickitti \
    --save_dir /home/johnl/data/semantickitti/mono_depth \
    -p local::$CKPT_PATH
}
exeFunc
