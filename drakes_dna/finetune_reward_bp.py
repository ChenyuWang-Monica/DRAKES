# direct reward backpropagation
import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
from utils import str2bool, set_seed


def fine_tune(new_model,  new_model_y, new_model_y_eval, old_model, args, eps=1e-5):

    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    new_model.config.finetuning.truncate_steps = args.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    dt = (1 - eps) / args.total_num_steps
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    batch_losses = []
    batch_rewards = []

    for epoch_num in range(args.num_epochs):
        rewards = []
        rewards_eval = []
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_model.train()
        for _step in range(args.num_accum_steps):
            sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp) # [bsz, seqlen, 4]
            sample2 = torch.transpose(sample, 1, 2)
            preds = new_model_y(sample2).squeeze(-1) # [bsz, 3]
            reward = preds[:, 0]

            sample_argmax = torch.argmax(sample, 2)
            sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes= 4)
            sample_argmax = torch.transpose(sample_argmax, 1, 2)

            preds_argmax = new_model_y(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
            rewards.append(reward_argmax.detach().cpu().numpy())
            
            preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
            reward_argmax_eval = preds_eval[:, 0]
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            
            total_kl = []
            
            # calculate the KL divergence
            for random_t in range(args.total_num_steps):
                if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
                    continue
                last_x = last_x_list[random_t] # [bsz, seqlen, 5]
                condt = condt_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                copy_flag = copy_flag_list[random_t] # [bsz, seqlen, 1]
                log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
                log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]

                p_x0 = log_p_x0.exp() # [bsz, seqlen, 4]
                p_x0_old = log_p_x0_old.exp()

                kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0,0,0]
                kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2)) # [bsz]
                total_kl.append(kl_div)

            if epoch_num < args.alpha_schedule_warmup:
                # linear warmup
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha

            kl_loss = torch.stack(total_kl, 1).sum(1).mean()
            reward_loss = - torch.mean(reward)
            loss = reward_loss + kl_loss * current_alpha
            loss = loss / args.num_accum_steps
            
            loss.backward()
            if (_step + 1) % args.num_accum_steps == 0: # Gradient accumulation
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            batch_losses.append(loss.cpu().detach().numpy())
            batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            reward_losses.append(reward_loss.cpu().detach().numpy())
            kl_losses.append(kl_loss.cpu().detach().numpy())
        
        rewards = np.array(rewards)
        rewards_eval = np.array(rewards_eval)
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        kl_losses = np.array(kl_losses)

        print("Epoch %d"%epoch_num, "Mean reward %f"%np.mean(rewards), "Mean reward eval %f"%np.mean(rewards_eval), 
        "Mean grad norm %f"%tot_grad_norm, "Mean loss %f"%np.mean(losses), "Mean reward loss %f"%np.mean(reward_losses), "Mean kl loss %f"%np.mean(kl_losses))
        if args.name != 'debug':
            wandb.log({"epoch": epoch_num, "mean_reward": np.mean(rewards), "mean_reward_eval": np.mean(rewards_eval), 
            "mean_grad_norm": tot_grad_norm, "mean_loss": np.mean(losses), "mean reward loss": np.mean(reward_losses), "mean kl loss": np.mean(kl_losses)})
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch_num} Mean reward {np.mean(rewards)} Mean reward eval {np.mean(rewards_eval)} Mean grad norm {tot_grad_norm} Mean loss {np.mean(losses)} Mean reward loss {np.mean(reward_losses)} Mean kl loss {np.mean(kl_losses)}\n")
        
        if (epoch_num+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
    
    if args.name != 'debug':
        wandb.finish()

    return batch_losses

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='/data/scratch/wangchy/seqft/')
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=1000)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--name', type=str, default='debug')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=50)
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()
print(args)

# pretrained model path
CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
log_base_dir = os.path.join(args.base_path, 'mdlm/reward_bp_results_final')

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# initialize a log file
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = f'alpha{args.alpha}_accum{args.num_accum_steps}_bsz{args.batch_size}_truncate{args.truncate_steps}_temp{args.gumbel_temp}_clip{args.gradnorm_clip}_{args.name}_{curr_time}'
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='reward_bp_final', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# Initialize the model
new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
reward_model.eval()
reward_model_eval.eval()

fine_tune(new_model, reward_model, reward_model_eval, old_model, args)
