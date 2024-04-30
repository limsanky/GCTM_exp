#!/bin/bash

AVAILABLE_GPU_INDICES='4'

CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU_INDICES} python ../../flow_matching/train_sfm.py \
--datasets 'cifar10' 'gaussian' \
--data_roots '/root/data/cifar10' '/root/data/gaussian' \
--base_dir '/root/code/results/sfm/0501/cifar10_sfm/' \
--size 32 \
--X1_eps_std 0.0 \
--vars 0.25 1.0 0.0 \
--coupling 'independent' \
--param 'sfm' \
--coupling_bs 64 \
--disc_steps 1280 \
--init_steps 10 \
--double_iter 50000 \
--discretization 'edm_n2i' \
--smin 0.002 \
--smax 80.0 \
--edm_rho 7 \
--ODE_N 1 \
--bs 64 \
--lr 2e-4 \
--lmda_CTM 1.0 \
--ctm_distance 'ph' \
--ema_decay 0.999 \
--n_grad_accum 1 \
--nc 3 \
--model_channels 128 \
--num_blocks 4 \
--dropout 0.1 \
--offline \
--n_FID 5000 \
--t_sm_dists 'lognormal' \
# --compare_zero