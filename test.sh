#! /bin/bash

export SEG=hanlp
export DATA_PATH=word_seg_$SEG

python2 split_pos_neg_new.py new_test $SEG # output_test.txt => new_test_neg_data
python2 make_format.py $DATA_PATH/new_test_neg_data 0  > $DATA_PATH/new_test_neg_data.fat
cat  $DATA_PATH/new_test*.fat  > $DATA_PATH/new_test_data
python2 entity_model/predict_new.py --data_path $DATA_PATH  > $DATA_PATH/new_score

python2 entity_model/get_result.py $DATA_PATH