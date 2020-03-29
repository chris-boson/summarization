export CUDA_VISIBLE_DEVICES=0,1,2,3

HOME_DIR=/home/lambda/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')
MODEL_TYPE=conditional_generation # language_model, encoder_decoder, conditional_generation
DATASET=tifu # tifu, cnn_dm
ENCODER=t5-small # transfo-xl-wt103, t5-small, gpt2, bert-base-uncased
DECODER=None

python -m trainer.main \
    --gpus 0,1,2,3 \
    --distributed_backend dp \
    --precision 32 \
    --home_dir $HOME_DIR \
    --dataset $DATASET \
    --name $DATASET/$MODEL_TYPE/$NOW \
    --accumulate_grad_batches 1 \
    --max_epochs 2 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --test_percentage 0.05 \
    --max_documents 1000 \
    --model_type $MODEL_TYPE \
    --max_tokens 512 \
    --encoder $ENCODER \
    --decoder $DECODER \
    --num_beams 3 \
    --max_length 40 \
    --repetition_penalty 3.0 \

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