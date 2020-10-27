export CUDA_VISIBLE_DEVICES=0,1,2,3

HOME_DIR=~/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')

MODEL_TYPE=conditional_generation # language_model, encoder_decoder, conditional_generation
ENCODER=t5-small # transfo-xl-wt103, t5-small, gpt2, bert-base-uncased
DECODER=None # If None use same as encoder

DATASET=ackchyually # tifu, cnn_dm


python -m trainer.main \
    --gpus 0 \
    --distributed_backend dp \
    --precision 32 \
    --home_dir $HOME_DIR \
    --model_type $MODEL_TYPE \
    --encoder $ENCODER \
    --decoder $DECODER \
    --dataset $DATASET \
    --name $DATASET/$MODEL_TYPE/${ENCODER}_${DECODER}/$NOW \
    --accumulate_grad_batches 1 \
    --max_epochs 2 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --test_percentage 0.1 \
    --max_documents 100 \
    --max_tokens 512 \
    --num_beams 1 \
    --max_length 40 \
    --repetition_penalty 3.0 \
