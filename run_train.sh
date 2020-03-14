HOME_DIR=/home/lambda/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')
MODEL_TYPE=gpt2 #gpt2, encoder_decoder

python -W ignore -m trainer.main \
    --home_dir $HOME_DIR \
    --name $MODEL_TYPE/$NOW \
    --gpus 0,1,2,3 \
    --train_batch_size 4 \
    --test_percentage 0.1 \
    --max_epochs 10 \
    --max_documents 100 \
    --model_type $MODEL_TYPE \
    --encoder gpt2 \
    --decoder gpt2 \
