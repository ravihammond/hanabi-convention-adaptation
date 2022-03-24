#!/bin/bash

python train.py \
       --save_dir exps/test \
       --num_thread 1 \
       --num_game_per_thread 1 \
       --partner ../models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw \
       --save_checkpoints 10 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 2254257 \
       --batchsize 1 \
       --burn_in_frames 1 \
       --replay_buffer_size 100000 \
       --epoch_len 1 \
       --num_epoch 1 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2,cuda:3 \
       --num_lstm_layer 2 \
       --min_t 0.01 \
       --max_t 0.1 \
       --load_model None \
       --net publ-lstm \
       --convention conventions/hint_red_play_0.json \
       --convention_act_override 1 \

