HOME_DIR=/home/lambda/projects/summarization
NOW=$(date +'%Y_%m_%d__%H_%M_%Z')

python -W ignore -m trainer.main \
    --home_dir $HOME_DIR \
    --name gpt2/$NOW \
    --gpus 0,1,2,3 \
    --train_batch_size 4 \
    --test_percentage 0.1 \
    --max_epochs 10 \
    --max_documents 1000 \
