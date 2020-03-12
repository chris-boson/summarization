HOME_DIR=/home/lambda/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')

python -W ignore -m trainer.main \
    --home_dir $HOME_DIR \
    --name gpt2/$NOW \
    --gpus 0,1 \
    --train_batch_size 4 \
