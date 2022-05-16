
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0
# export CPU_NUM=10
# export NVIDIA_TF32_OVERRIDE=0

export PADDLE_TRAINERS_NUM=1

output_dir="./output/cylinder3d_auto_serial_gm16"
export PYTHONPATH=./../../../PaddleScience:$PYTHONPATH

mkdir -p $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export FLAGS_call_stack_level=2
# export GLOG_vmodule=operator=4
# export GLOG_v=5
export FLAGS_START_PORT=6687
# export FLAGS_cinn_use_new_fusion_pass=1

python -m paddle.distributed.launch \
    --log_dir ${output_dir} \
    --gpus=${CUDA_VISIBLE_DEVICES} \
    cylinder3d_unsteady_auto.py
