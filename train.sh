#! /bin/bash

export SEG=hanlp
export DATA_PATH=word_seg_$SEG

python2 split_pos_neg.py train $SEG
python2 split_pos_neg.py valid $SEG
python2 split_pos_neg.py test $SEG


python2 make_format.py $DATA_PATH/train_pos_data 1 > $DATA_PATH/train_pos_data.fat
python2 make_format.py $DATA_PATH/train_neg_data 0 > $DATA_PATH/train_neg_data.fat
python2 make_format.py $DATA_PATH/valid_pos_data 1 > $DATA_PATH/valid_pos_data.fat
python2 make_format.py $DATA_PATH/valid_neg_data 0 > $DATA_PATH/valid_neg_data.fat
python2 make_format.py $DATA_PATH/test_pos_data 1  > $DATA_PATH/test_pos_data.fat
python2 make_format.py $DATA_PATH/test_neg_data 0  > $DATA_PATH/test_neg_data.fat
 
cat  $DATA_PATH/train*.fat > $DATA_PATH/train_data
cat  $DATA_PATH/valid*.fat > $DATA_PATH/valid_data
cat  $DATA_PATH/test*.fat  > $DATA_PATH/test_data

mkdir -p $DATA_PATH/model

python2 entity_model/train.py --data_path $DATA_PATH 
python2 entity_model/predict.py --data_path $DATA_PATH  > $DATA_PATH/score

python2 entity_model/search_threshold.py $DATA_PATH

