export CUDA_VISIBLE_DEVICES=0,1,2,3

HOME_DIR=/home/lambda/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')
MODEL_TYPE=gpt2 # gpt2, encoder_decoder, bart
DATASET=tifu # tifu, cnn_dm

python -m trainer.main \
    --gpus 0,1,2,3 \
    --distributed_backend dp \
    --precision 32 \
    --home_dir $HOME_DIR \
    --dataset $DATASET \
    --name $DATASET/$MODEL_TYPE/$NOW \
    --accumulate_grad_batches 1 \
    --max_epochs 10 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --test_percentage 0.1 \
    --max_documents 100 \
    --model_type $MODEL_TYPE \
    --max_tokens 1024 \

# python -m torch.distributed.launch \
#     --nproc_per_node 4 \
#     trainer/main.py \
#         --home_dir $HOME_DIR \
#         --name $DATASET/$MODEL_TYPE/$NOW \
#         --gpus 0,1,2,3 \
#         --distributed_backend ddp \
#         --precision 32 \
#         --max_epochs 10 \
#         --train_batch_size 1 \
#         --eval_batch_size 1 \
#         --test_percentage 0.1 \
#         --max_documents 100 \
#         --model_type $MODEL_TYPE \
#         --max_tokens 1024 \