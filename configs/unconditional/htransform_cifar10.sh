# # CUDA_VISIBLE_DEVICES=0 python train_gctm_adam.py \
# CUDA_VISIBLE_DEVICES=0 python ../../train_gctm.py \
# --datasets 'cifar10' 'gaussian' \
# --base_dir 'root/results/cifar10/cifar10_norm' \
# --size 32 \
# --X1_eps_std 0.0 \
# --vars 0.25 1.0 0.0 \
# --coupling 'independent' \
# --coupling_bs 128 \
# --disc_steps 1024 \
# --init_steps 4 \
# --double_iter 50000 \
# --discretization 'edm_n2i' \
# --smin 0.002 \
# --smax 80.0 \
# --edm_rho 7 \
# --t_sm_dist 'lognormal' \
# --ODE_N 1 \
# --param 'LIN' \
# --bs 128 \
# --lr 2e-4 \
# --rho 0.001 \
# --lmda_CTM 1.0 \
# --ctm_distance 'ph' \
# --ema_decay 0.999 \
# --n_grad_accum 1 \
# --nc 3 \
# --model_channels 128 \
# --num_blocks 4 \
# --dropout 0.1 \
# --offline

# CUDA_VISIBLE_DEVICES=0 python train_gctm_adam.py \
CUDA_VISIBLE_DEVICES=0 python ../../train_gctm_htransform.py \
--datasets 'cifar10' 'gaussian' \
--data_roots '/root/data/cifar10' '/root/data/gaussian' \
--base_dir '/root/code/results/0427/htransform/cifar10_uncond/' \
--size 32 \
--X1_eps_std 0.05 \
--vars 0.25 1.0 0.0 \
--coupling 'htransform' \
--coupling_bs 128 \
--disc_steps 1280 \
--init_steps 10 \
--double_iter 50000 \
--discretization 'edm_n2i' \
--smin 0.002 \
--smax 80.0 \
--edm_rho 7 \
--ODE_N 1 \
--bs 128 \
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
--param 'hTransform' \
# --compare_zero