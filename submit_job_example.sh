#!/bin/bash
#SBATCH --job-name=bert_amazon_T5ForSequenceClassification
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#initial_checkpoint="checkpoints/bart_large/"
#output_prefix="checkpoints/bart_imdb_layer"
#train_file="dataset/sentiment/original_imdb/tc_train.csv"
#valid_file="dataset/sentiment/original_imdb/tc_valid.csv"
#model_class="BartForDisentangledRepresentation_incremental"
#random_seed=1000
#num_train_epochs=100
#
#for hidden_state_layer in {12..1}; do
#    if [ $hidden_state_layer -eq 12 ]; then
#        model_path=$initial_checkpoint
#    else
#        prev_layer=$((hidden_state_layer + 1))
#        model_path=${output_prefix}${prev_layer}_${model_class}_${random_seed}/
#    fi
#
#    output_dir=${output_prefix}${hidden_state_layer}_${model_class}_${random_seed}
#
#    python run_text_disentangled_classification.py \
#        --train_file $train_file \
#        --validation_file $valid_file \
#        --model_name_or_path $model_path \
#        --num_train_epochs $num_train_epochs \
#        --hidden_state_layer $hidden_state_layer \
#        --output_dir $output_dir \
#        --model_class $model_class \
#        --seed $random_seed
#done

#python interact_chatgpt.py
#python run_text_classification.py --train_file dataset/SocialDial2.0/norm_category_synthetic_human_balance_train.csv --validation_file dataset/SocialDial2.0/norm_category_valid.csv --model_name_or_path xlm-roberta-base --num_train_epochs 10 --learning_rate 1e-5 --output_dir checkpoints_socialDial2.0/roberta_norm_category_syntheticHuman_1e-5
python run_text_disentangled_classification.py --train_file dataset/sentiment/amazon_polarity/tc_train.csv --validation_file dataset/sentiment/amazon_polarity/tc_valid.csv --model_name_or_path checkpoints/t5_large/ --num_train_epochs 1  --output_dir checkpoints/t5_amazon_T5ForSequenceClassification --model_class T5ForSequenceClassification --seed 415
#python run_text_domain_classification.py --train_file dataset/sentiment/domain_classification/amazon_yahoo_test.csv --validation_file dataset/sentiment/domain_classification/amazon_yahoo_test.csv --model_name_or_path checkpoints_domain/amazon_yahoo_domain/ --num_train_epochs 1 --output_dir checkpoints_domain/out --only_evaluation
#python run_text_disentangled_classification_socialDial.py --train_file dataset/sentiment/amazon_review_zh/train/train_violate_cls.csv --validation_file dataset/SocialDial/Human_written_Dialogue/norm_violation_cls_test.csv --model_name_or_path checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_CLS_attnMultihead_ChatYuan_merge/ --backbone_model checkpoints_socialDial/ChatYuan_large_v1/ --num_train_epochs 5 --per_device_train_batch_size 32 --output_dir checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_NV_CLS_attnMultihead_ChatYuan_merge/ --cls_head norm_violate
#python run_text_disentangled_classification_binaryclass.py --train_file dataset/sentiment/filter_yahoo/train/tc_train.csv --validation_file dataset/sentiment/filter_yahoo/valid/tc_valid.csv --model_name_or_path checkpoints/bert_large --backbone_model checkpoints/bert_large/ --num_train_epochs 1000 --per_device_train_batch_size 32 --output_dir checkpoints/sequenceCLS_sparseAttnJoint_bert_yahoo --model_class BertForTokenAttentionSparseCLSJoint
#python run_text_disentangled_classification_multiclass.py --train_file dataset/ag_news/train/tc_description_train.csv --validation_file dataset/ag_news/validation/tc_description_validation.csv --model_name_or_path checkpoints/bart_large/ --backbone_model checkpoints/bart_large/ --num_train_epochs 100 --hidden_state_layer 12 --output_dir checkpoints_ag_news/sequenceCLS_sparseAttnJointMultiClass_bart_description_incremental --model_class BartForTokenAttentionSparseCLSJointMultiClass_incremental