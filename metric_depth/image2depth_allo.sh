CKPT_PATH="/home/johnl/misc/depth_anything/depth_anything_allo.pt"
set -e
exeFunc(){
    CUDA_VISIBLE_DEVICES=0 python3 predict.py \
    -m zoedepth -d allo \
    --save_dir /home/johnl/data/allo_3d/mono_depth \
    --save_images \
    --test_all \
    -p local::$CKPT_PATH
}
exeFunc
