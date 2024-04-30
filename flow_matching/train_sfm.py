from subprocess import call
import sys
import os
import shutil

from utils import create_dir, backup_dir, eval_FID, viz_img, delete_all_but_N_files
from discretizations import get_discretization
from averagemeter import AverageMeter
from distances import get_distance
from solvers import get_solver
from networks import SongUNet
from data import Sampler

import torch.optim as optim
import numpy as np
import pprint

import argparse
import torch
import copy
import math
import time
import os

from torch_utils import logger
import nvidia_smi
import gc

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def calculate_adaptive_weight(loss_top, loss_bottom, last_layer_weights=None):
    loss_top_grad = torch.autograd.grad(loss_top, last_layer_weights, retain_graph=True)[0]
    loss_bottom_grad = torch.autograd.grad(loss_bottom, last_layer_weights, retain_graph=True)[0]
    
    d_weight = torch.norm(loss_top_grad) / (torch.norm(loss_bottom_grad) + 1e-4)
    
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight.item()

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        return value
    return weight

def improved_loss_weighting(sigmas: torch.Tensor) -> torch.Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])

def save_ckpt(X0_eval, X1_eval, net, net_ema, opt_CTM, avgmeter, best_FID, ckpt_dir, idx, best=False):
    
    ckpt = {
        'X0_eval': X0_eval,
        'X1_eval': X1_eval,
        'net': net.state_dict(),
        'net_ema' : net_ema.state_dict(),
        'opt_CTM' : opt_CTM.state_dict(),
        'avgmeter': avgmeter.state_dict(),
        'best_FID' : best_FID
    }
    
    if best:
        torch.save(ckpt, os.path.join(ckpt_dir, 'idx_0_best.pt'))
    else:
        torch.save(ckpt, os.path.join(ckpt_dir, 'idx_{}_curr.pt'.format(idx)))

def train(args, datasets, data_roots, X1_eps_std, vars, coupling, lmda_CTM, solver, ctm_distance, compare_zero, size, discretization, smin, smax, edm_rho,
          t_sm_dists, disc_steps, init_steps, ODE_N, bs, coupling_bs, lr, ema_decay, n_grad_accum, offline, double_iter, t_ctm_dists,
          nc, model_channels, num_blocks, dropout, param, v_iter, s_iter, b_iter, FID_iter, FID_bs, n_FID, n_viz, n_save, base_dir, ckpt_name):
    
    CORRECT_EXP = (coupling == 'independent' and param == 'sfm')
    
    # Mistakes are fixed (by sanky).
    USE_FIXED_VERSION = False
    USE_COMBINED_LOSS = True
    USE_BF16 = False
    if USE_BF16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
    # print_gpu_usage('Before everything')
    
    # print("\n\n\nonce more!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n")
    size = max(size, 32)
    sampler = Sampler(datasets, data_roots, nc, size, X1_eps_std, coupling, coupling_bs, bs)
    disc = get_discretization(discretization,disc_steps,smin=smin,smax=smax,rho=edm_rho,t_sm_dists=t_sm_dists,t_ctm_dists=t_ctm_dists)
    ctm_dist, l2_loss = get_distance(ctm_distance), get_distance('l2')
    # ddbm_dist = get_distance('ph')
    
    solver = get_solver(solver, disc)

    vars[1] += X1_eps_std**2

    net: torch.nn.Module = SongUNet(vars=vars, param=param, discretization=disc, img_resolution=size, in_channels=nc, out_channels=nc,
                   num_blocks=num_blocks, dropout=dropout, model_channels=model_channels,
                #    encoder_type='residual',
                ).cuda()
    
    if USE_BF16:
        net.bfloat16()
    
    # opt_CTM = optim.Adam(net.parameters(), lr=lr)
    # opt_CTM = optim.RAdam(net.parameters(), lr=4e-4, weight_decay=0.0)
    opt_CTM = optim.RAdam(net.parameters(), lr=2e-4, weight_decay=0.0)
    net_ema: torch.nn.Module = copy.deepcopy(net)
    
    if USE_BF16:
        net_ema.bfloat16()
    
    # avgmeter = AverageMeter(window=125,
    #                         loss_names=['DSM Loss', 'CTM Loss', 'FM Loss',
    #                                     'DSM Weight', 'FM Weight',
    #                                     'FID'],
    #                         yscales=['log', 'log', 'log',
    #                                  'linear', 'linear',
    #                                  'linear'])
    
    avgmeter = AverageMeter(window=125,
                            loss_names=['DBSM Loss', 'CTM Loss', 
                                        'DBSM Weight',
                                        'FID'],
                            yscales=['log', 'log',
                                     'linear', 
                                     'linear'])

    loss_dir = os.path.join(base_dir, 'losses')
    sample_B_dir = os.path.join(base_dir, 'samples_B')
    sample_F_dir = os.path.join(base_dir, 'samples_F')
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    # X0_stat_path = os.path.join('FID_stats', '{}_{}.npz'.format(datasets[0],size))
    X0_stat_path = os.path.join('/root/code/FID_stats', '{}_{}.npz'.format(datasets[0],size))
    # X0_stat_path = os.path.join(base_dir, 'FID_stats0', '{}_{}.npz'.format(datasets[0], size))

    if ckpt_name:
        print('\nLoading state from [{}]\n'.format(ckpt_name))
        ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name))
        X0_eval = ckpt['X0_eval'].cuda()
        X1_eval = ckpt['X1_eval'].cuda()
        net.load_state_dict(ckpt['net'])
        opt_CTM.load_state_dict(ckpt['opt_CTM'])
        net_ema.load_state_dict(ckpt['net_ema'])
        avgmeter.load_state_dict(ckpt['avgmeter'])
        # loss_DSM = avgmeter.losses['DSM Loss'][-1]
        loss_DSM = avgmeter.losses['DBSM Loss'][-1]
        loss_CTM = avgmeter.losses['CTM Loss'][-1]
        # loss_FM = avgmeter.losses['FM Loss'][-1]
        best_FID = ckpt['best_FID']
    else:
        X0_eval = torch.cat([sampler.sample_X0() for _ in range(math.ceil(n_viz/bs))], dim=0)[:n_viz]
        X1_eval = torch.cat([sampler.sample_X1() for _ in range(math.ceil(n_viz/bs))], dim=0)[:n_viz]
        best_FID = 10000
        create_dir(base_dir, prompt=True)
        create_dir(loss_dir)
        create_dir(sample_B_dir)
        create_dir(sample_F_dir)
        create_dir(ckpt_dir)
        # create_dir(os.path.join(base_dir, 'FID_stats0'))
        # create_dir(X0_stat_path)
    
    logger.configure(args, dir=base_dir)
    
    # Evaluate initial FID and visualize X1 samples
    t0 = disc.get_ts(disc_steps)[0]
    t1 = disc.get_ts(disc_steps)[-1]

    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float32):
    curr_FID = eval_FID(lambda:sampler.sample_X1(), X0_stat_path, t1, t0, net_ema, n_FID, sample_B_dir, fid_bs=FID_bs, verbose=True)
    print_gpu_usage('After computing DM FIDs')
    viz_img(X1_eval, t1, t1, net_ema, sample_B_dir, 0)
    viz_img(X0_eval, t0, t0, net_ema, sample_F_dir, 0)
    
    
    def get_scaling_constants(sigma_t_squared, sigma_T_squared, alpha_t=1, alpha_T=1):
        assert (alpha_t is not None), f'alpha_t cannot be None!'
        assert (alpha_T is not None), f'alpha_T cannot be None!'
        assert (sigma_t_squared is not None), f'sigma_t_squared cannot be None!'
        assert (sigma_T_squared is not None), f'sigma_T_squared cannot be None!'
        
        # at = αt/αT * SNRT/SNRt; bt = αt(1 − SNRT/SNRt); ct = (σt^2)(1 − SNRT/SNRt)
        
        SNR_t = np.square(alpha_t) / sigma_t_squared
        SNR_T = np.square(alpha_T) / sigma_T_squared
        
        a_t = (alpha_t / alpha_T) * (SNR_T / SNR_t)
        b_t = alpha_t * (1 - (SNR_T / SNR_t))
        # c_t = sigma_t_squared * (1 - (SNR_T / SNR_t))
        c_t = np.reshape(sigma_t_squared, newshape=(-1,1,1,1))
        
        return a_t.reshape(-1,1,1,1), b_t.reshape(-1,1,1,1), c_t
        

    # Training loop
    sub_steps_history = []
    seed = int(time.time())
    
    print_gpu_usage('Before Emptying Cache')
    clear_cache()
    print_gpu_usage('After Emptying Cache; Before Training')
        
    while True:
        
        if double_iter is not None:
            sub_steps = min(disc_steps, init_steps*2**(avgmeter.idx//double_iter))
        else:
            sub_steps = init_steps
        
        # print('sub_steps:',sub_steps)
        assert (sub_steps <= 1280)
        if len(sub_steps_history) == 0:
            sub_steps_history.append(sub_steps)
            
        if sub_steps_history[-1] != sub_steps:
            assert sub_steps_history[-1] < sub_steps, 'There is some error in the sub_steps history list.'
            sub_steps_history.append(sub_steps)
            print(f'Number of Iterations has doubled, from {sub_steps_history[-2]} -> {sub_steps_history[-1]}.')
        
        # Freeze EMA Model (4/19)
        net_ema.requires_grad_(False)
        opt_CTM.zero_grad()
        
        for accum_idx in range(n_grad_accum):
            # Sample data
            X0, X1 = sampler.sample_joint()
            
            t_sm_idx, t_sm = disc.sample_sm_times(bs, disc_steps)
            t_idx, s_idx, u_idx, v_idx, t, s, u, v = disc.sample_ctm_times(bs, sub_steps)
            
            assert (CORRECT_EXP), f"Wrong experiment!\nShould be {param} w/ {coupling} coupling."
            
            # For SFM w/ Independent Coupling: 
            #               a_t = αt/αT ∗ SNRT/SNRt, 
            #               b_t = αt(1 − SNRT/SNRt), 
            #               c_t = (σt^2)(1 − SNRT/SNRt)
            
            a_t, b_t, c_t = get_scaling_constants(sigma_t_squared=t, sigma_T_squared=1)
            noise = torch.randn_like(X0, device=X0.device)
            X_t = (a_t * X1) + (b_t * X0) + (np.sqrt(c_t) * noise)
            
            a_t_sm, b_t_sm, c_t_sm = get_scaling_constants(sigma_t_squared=t_sm, sigma_T_squared=1)
            noise2 = torch.randn_like(X1, device=X1.device)
            X_t_sm = (a_t_sm * X1) + (b_t_sm * X0) + (np.sqrt(c_t_sm) * noise2)
            
            # noise = torch.randn_like(X0, device=X0.device)
            # Xt = (1 - t).reshape(-1,1,1,1) * X0 + t.reshape(-1,1,1,1) * X1 + ((1 - t.reshape(-1,1,1,1)) ** 0.5) * eps * t.reshape(-1,1,1,1) # < Changeed on: (4/27)
                        
            # # xt_for_ot = Xt 
            # ut_for_ot = X1 - X0
            
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if USE_BF16 else torch.float32):
            # Calculate CTM Loss
            with torch.no_grad():
                if offline:
                    net_ema.eval()
                    X_u_solved = solver.solve(X_t, t_idx, u_idx, net_ema, sub_steps, ODE_N)
                else:
                    net.train()
                    X_u_solved = solver.solve(X_t, t_idx, u_idx, net, sub_steps, ODE_N, seed)
                
            if USE_FIXED_VERSION:
                # net.train()
                net_ema.eval()
                torch.manual_seed(seed)
                X_s_real = net_ema(X_u_solved, u, s)[0] # <- Fixed.
            else:
                net.train()
                torch.manual_seed(seed)
                X_s_real = net(X_u_solved, u, s)[0] # <- Mistake; should've used the stop-grad model.
        
            # net_ema.eval()
            # X_real = net_ema(Xs_real, s, t0*torch.zeros_like(s)) if compare_zero else Xs_real
            if compare_zero:
                net_ema.eval()
                X_real = net_ema(X_s_real, s, torch.zeros_like(s)*t0)
            else:
                X_real = X_s_real
            
            net.train()
            net_ema.eval()
            torch.manual_seed(seed)
            
            X_s_fake, cout = net(X_t, t, s)
            
            # X_fake = net_ema(Xs_fake,s,t0*torch.zeros_like(s)) if compare_zero else Xs_fake
            if compare_zero:
                G_theta = net_ema(X_s_fake, s, torch.zeros_like(s)*t0)
            else:
                G_theta = X_s_fake
            
            # vt_for_ot = net()
            
            if USE_COMBINED_LOSS:
                
                # Calculate CTM Loss
                loss_CTM = lmda_CTM * ctm_dist(G_theta, X_real, cout * (1 - s/t)) / (n_grad_accum * bs)
                # loss_CTM.backward()
                seed += 1

                # Calculate DSM Loss
                net.train()
                g_theta, cout = net(X_t_sm, t_sm, t_sm, return_g_theta=True)
                
                loss_DBSM = l2_loss(X0, g_theta, cout) / (n_grad_accum*bs)
                
                weight_DBSM = calculate_adaptive_weight(loss_CTM, loss_DBSM, net.dec['32x32_aux_conv'].weight)
                balance_weight_DBSM = adopt_weight(weight_DBSM, global_step=avgmeter.idx, threshold=0, value=1.)
                
                loss = loss_CTM + (balance_weight_DBSM * loss_DBSM) # + (balance_weight_FM * loss_FM)
                
                # loss_DSM = l2_loss(X0, g_theta, cout) / (n_grad_accum*bs)
                
                # weight_DSM = calculate_adaptive_weight(loss_CTM, loss_DSM, net.dec['32x32_aux_conv'].weight)
                # balance_weight_DSM = adopt_weight(weight_DSM, global_step=avgmeter.idx, threshold=0, value=1.)                
                # balance_weight_DSM = 1
                
                # print("weight DSM:", weight_DSM.item(), 'balance weight dsm', balance_weight_DSM.item())
                # weights = improved_loss_weighting() # 1 / (t-u)
                
                # loss_DSM.backward()
                
                # loss_FM = l2_loss(X1 - X0, aux_from_unet)
                
                # weight_FM = calculate_adaptive_weight(loss_CTM, loss_FM, net.dec['32x32_aux_conv'].weight)
                # balance_weight_FM = adopt_weight(weight_FM, global_step=avgmeter.idx, threshold=0, value=1.)
                
                # loss = loss_CTM + (balance_weight_DSM * loss_DSM) + (balance_weight_FM * loss_FM)
                loss.backward()
            else:
                
                # Calculate CTM Loss
                loss_CTM = lmda_CTM * ctm_dist(G_theta, X_real, weight=cout * (1 - s/t)) / (n_grad_accum * bs)
                # weights = improved_loss_weighting() # 1 / (t-u)
                loss_CTM.backward()
                seed += 1

                # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Calculate DSM Loss
                net.train()
                g_theta, cout = net(X_t_sm, t_sm, t_sm, return_g_theta=True)
                loss_DSM = l2_loss(X0, g_theta, cout) / (n_grad_accum*bs)
                
                loss_DSM.backward()
                
            if accum_idx == n_grad_accum-1:
                opt_CTM.step()
                # print_gpu_usage('After one step training; Before emptying cache')
                # clear_cache()
                # print_gpu_usage('After emptying cache')

        # EMA update
        if double_iter is not None:
            ema_list = [0.999, 0.9999, 0.99993]
            # ema_decay_curr = ema_list[min(int(avgmeter.idx//double_iter), 2)]
            # ema_decay_curr = 0. # Following iCT
            # ema_decay_curr = 0.99993
            ema_decay_curr = 0.999
        else:
            ema_decay_curr = ema_decay
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                p_ema.data = ema_decay_curr * p_ema + (1 - ema_decay_curr) * p

        # Loss tracker update
        avgmeter.update({
            # 'DSM Loss' : loss_DSM.item()*n_grad_accum,
            'DBSM Loss' : loss_DBSM.item()*n_grad_accum,
            'CTM Loss' : loss_CTM.item()*n_grad_accum,
            'FID'      : curr_FID,
            # 'DSM Weight': balance_weight_DSM,
            'DBSM Weight': balance_weight_DBSM,
            })
        
        # Loss and sample visualization
        if avgmeter.idx % v_iter == 0:
            # print(avgmeter)
            logger.log(avgmeter)
            avgmeter.plot_losses(os.path.join(loss_dir, 'losses.jpg'), nrows=1)
            viz_img(X1_eval, t1, t0, net_ema, sample_B_dir, None)

        # Saving checkpoint
        if avgmeter.idx % s_iter == 0:
            print('\nSaving checkpoint at [{}], Best FID : {:.2f}\n'.format(ckpt_dir,best_FID))
            save_ckpt(X0_eval, X1_eval, net, net_ema, opt_CTM, avgmeter, best_FID, ckpt_dir, avgmeter.idx)
            delete_all_but_N_files(ckpt_dir, lambda x : int(x.split('_')[1]), n_save, 'best')
            viz_img(X1_eval, t1, t0, net_ema, sample_B_dir, avgmeter.idx)
        
        # Saving backup checkpoint
        if avgmeter.idx % b_iter == 0:
            print('\nSaving backup checkpoint at [{}]\n'.format(base_dir,best_FID))
            save_ckpt(X0_eval, X1_eval, net, net_ema, opt_CTM, avgmeter, best_FID, base_dir, avgmeter.idx)
        
        # Evaluating Quick FID
        if avgmeter.idx % FID_iter == 0:
            # if not ckpt_name:
            #     create_dir(os.path.join(base_dir, f'FID_stats{avgmeter.idx//FID_iter}'))
            # X0_stat_path_new = os.path.join(base_dir, f'FID_stats{avgmeter.idx//FID_iter}', '{}_{}.npz'.format(datasets[0], size))
            # curr_FID = eval_FID(lambda:sampler.sample_X1(), X0_stat_path_new, t1, t0, net_ema, n_FID, sample_B_dir, fid_bs=FID_bs, verbose=True)
            curr_FID = eval_FID(lambda: sampler.sample_X1(), X0_stat_path, t1, t0, net_ema, n_FID, sample_B_dir, fid_bs=FID_bs, verbose=True)
            if curr_FID < best_FID:
                best_FID = curr_FID
                save_ckpt(X0_eval, X1_eval, net, net_ema, opt_CTM, avgmeter, best_FID, ckpt_dir, avgmeter.idx, best=True)
        
        if avgmeter.idx % FID_iter * 10 == 0:
            # call(['chmod', '777', '-R', args.save_dir])
            call(['chmod', '777', '-R', '/root/code/'])
            # print('# Save dir: ', args.save_dir, '\n')
        
        if avgmeter.idx % 5000 == 0:
            print_gpu_usage(f'Before emptying cache @ iteration {avgmeter.idx}:')
            clear_cache()
            print_gpu_usage(f'After emptying cache @ iteration {avgmeter.idx}:')
        

nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
def print_gpu_usage(prefix=''):
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            
            usage = f"{prefix} |Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu / 100.0:3.1%} | gpu-mem: {util.memory / 100.0:3.1%} |"
            
            logger.log(usage)


def main():
    
    parser = argparse.ArgumentParser()

    # Basic experiment settings
    parser.add_argument('--datasets', type=str, nargs='+', default=['cifar10','gaussian'])
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data','../data'])
    parser.add_argument('--base_dir', type=str, default='results/cifar10')
    parser.add_argument('--ckpt_name', type=str, default=None)

    # p(X0,X1) settings
    # inverse tasks = {'sr4x-pool', 'sr4x-bicubic', 'inpaint-center', 'inpaint-random', 'blur-uni', 'blur-gauss'}
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--X1_eps_std', type=float, default=0.0)
    parser.add_argument('--vars', type=float, nargs='+', default=[0.25,1.0,0.0])
    parser.add_argument('--coupling', type=str, default='independent')
    parser.add_argument('--coupling_bs', type=int, default=64)

    # ODE settings
    parser.add_argument('--disc_steps', type=int, default=1024)
    parser.add_argument('--init_steps', type=int, default=8)
    parser.add_argument('--double_iter', type=int, default=None)
    parser.add_argument('--solver', type=str, default='heun')
    parser.add_argument('--discretization', type=str, default='edm_n2i')
    parser.add_argument('--smin', type=float, default=0.002)
    parser.add_argument('--smax', type=float, default=80.0)
    parser.add_argument('--edm_rho', type=int, default=7)
    parser.add_argument('--t_sm_dists', type=str, nargs='+', default=[])
    parser.add_argument('--t_ctm_dists', type=float, nargs='+', default=[1.2,2])
    parser.add_argument('--param', type=str, default='LIN')
    parser.add_argument('--ODE_N', type=int, default=1)

    # Optimization settings
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lmda_CTM', type=float, default=0.1)
    parser.add_argument('--ctm_distance', type=str, default='l1')
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--n_grad_accum', type=int, default=1)
    parser.add_argument('--compare_zero', action='store_true')
    parser.add_argument('--offline', action='store_true')

    # Model settings
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Evaluation settings
    parser.add_argument('--v_iter', type=int, default=25)
    parser.add_argument('--s_iter', type=int, default=250)
    parser.add_argument('--b_iter', type=int, default=25000)
    parser.add_argument('--FID_iter', type=int, default=250)
    parser.add_argument('--n_FID', type=int, default=5000)
    parser.add_argument('--FID_bs', type=int, default=500)
    parser.add_argument('--n_viz', type=int, default=100)
    parser.add_argument('--n_save', type=int, default=2)
    
    args = parser.parse_args()
    
    def print_args(**kwargs):
        print('\nTraining with settings :\n')
        pprint.pprint(kwargs)
    
    try:
        assert int(os.environ['WORLD_SIZE']) >= 1
    except:
        print('\n######################################################################################')
        print('### Execute shellscript file in the "./configs" directory; not a .py file directly ###')
        print('###                        Enter terminal command like below                       ###')
        print('###                                 $ . train_sfm.sh                               ###')
        print('######################################################################################\n')
        sys.exit()
        
    if int(os.environ['WORLD_SIZE']) > 1:
        # Multi GPUs
        print(f'Using multple GPUs.')
        # args.is_distributed = True
    else:
        # Or only one GPU
        print(f'Using one GPU.')
        # args.is_distributed = False
        
    print_args(**vars(args))
    train(args=args, **vars(args))

if __name__ == '__main__':
    main()