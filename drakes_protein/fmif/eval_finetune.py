import argparse
import os.path
import pickle
from protein_oracle.utils import str2bool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import pandas as pd
import numpy as np
import torch
import os
import shutil
import warnings
from torch.utils.data import DataLoader
import os.path
from protein_oracle.utils import set_seed
from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from protein_oracle.model_utils import ProteinMPNNOracle
from fmif.model_utils import ProteinMPNNFMIF
from fmif.fm_utils import Interpolant
from tqdm import tqdm
from multiflow.models import folding_model
from types import SimpleNamespace
import pyrosetta
pyrosetta.init(extra_options="-out:level 100")
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import *


def cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, args, item_idx, base_path):
    with torch.no_grad():
        results_list = []
        run_name = save_path.split('/')
        if run_name[-1] == '':
            run_name = run_name[-2]
        else:
            run_name = run_name[-1]
        
        sc_output_dir_base = os.path.join(base_path, 'sc_eval', f'{args.decoding}_{args.base_model}_{args.dps_scale}_{args.tds_alpha}_{args.seed}_{run_name}', 'sc_output', batch["protein_name"][0][:-4])
        sc_output_dir = os.path.join(sc_output_dir_base, 'true')
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
        # fold the ground truth sequence
        os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
        true_fasta = fasta.FastaFile()
        true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
        true_fasta['true_seq_1'] = true_detok_seq
        true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
        true_fasta.write(true_fasta_path)
        true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
        if os.path.exists(true_folded_dir):
            shutil.rmtree(true_folded_dir)
        os.makedirs(true_folded_dir, exist_ok=False)
        true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)
        true_folded_pdb_path = os.path.join(true_folded_dir, 'folded_true_seq_1.pdb')
        true_folded_pose = pyrosetta.pose_from_file(true_folded_pdb_path)

        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_folded_pose))
        packer.apply(true_folded_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_folded_pose)
        true_folded_relax_path = os.path.join(sc_output_dir, 'true_folded_relax', 'folded_true_relax_seq_1.pdb')
        os.makedirs(os.path.join(sc_output_dir, 'true_folded_relax'), exist_ok=True)
        true_folded_pose.dump_pdb(true_folded_relax_path)

        true_pose = pyrosetta.pose_from_file(the_pdb_path)
        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_pose))
        packer.apply(true_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_pose)

        os.makedirs(os.path.join(sc_output_dir, 'true_relax'), exist_ok=True)
        true_pose.dump_pdb(os.path.join(sc_output_dir, 'true_relax', 'true_relax_seq_1.pdb'))

        foldtrue_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, true_folded_pose)
    
        for _it, ssp in enumerate(S_sp):
            num = item_idx * 16 + _it
            sc_output_dir = os.path.join(sc_output_dir_base, f'{num}')
            os.makedirs(sc_output_dir, exist_ok=True)
            os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
            codesign_fasta = fasta.FastaFile()
            detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta['codesign_seq_1'] = detok_seq
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir, 'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)

            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            gen_folded_pdb_path = os.path.join(folded_dir, 'folded_codesign_seq_1.pdb')
            pose = pyrosetta.pose_from_file(gen_folded_pdb_path)
            scorefxn = pyrosetta.create_score_function("ref2015_cart")
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
            packer.apply(pose)
            relax = FastRelax()
            relax.set_scorefxn(scorefxn)
            relax.apply(pose)

            os.makedirs(os.path.join(sc_output_dir, 'folded_relax'), exist_ok=True)
            pose.dump_pdb(os.path.join(sc_output_dir, 'folded_relax', 'folded_codesign_relax_seq_1.pdb'))

            gen_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)
            gen_foldtrue_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_folded_pose, pose)
            seq_revovery = (S_sp[_it] == S[0]).float().mean().item()

            resultdf = pd.DataFrame(columns=['gen_true_bb_rmsd', 'gen_foldtrue_bb_rmsd', 'foldtrue_true_bb_rmsd', 'seq_recovery'])
            resultdf.loc[0] = [gen_true_bbrmsd, gen_foldtrue_bbrmsd, foldtrue_true_bbrmsd, seq_revovery]
            resultdf['seq'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            resultdf['true_seq'] = true_detok_seq
            resultdf['protein_name'] = batch['protein_name'][0]
            resultdf['WT_name'] = batch['WT_name'][0]
            resultdf['num'] = num
            resultdf['pdbpath'] = sc_output_dir
            results_list.append(resultdf)

    return results_list


def parse_df(results_df):
    avg_rmsd = results_df['bb_rmsd'].mean()
    success_rate = results_df['bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
    return avg_rmsd, success_rate, np.format_float_positional(avg_rmsd, unique=False, precision=3), np.format_float_positional(success_rate, unique=False, precision=3)


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--base_path", type=str, default="/data/scratch/wangchy/seqft/", help="base path for data and model") 
argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for") # 200
argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
argparser.add_argument("--batch_size", type=int, default=32, help="number of sequences for one batch") # 128
argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout") # TODO
argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
argparser.add_argument("--gradient_norm", type=float, default=1.0, help="clip gradient norm, set to negative to omit clipping")
argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--wd", type=float, default=1e-4)

argparser.add_argument("--min_t", type=float, default=1e-2)
argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
argparser.add_argument("--temp", type=float, default=0.1)
argparser.add_argument("--noise", type=float, default=1.0) # 20.0
argparser.add_argument("--interpolant_type", type=str, default='masking')
argparser.add_argument("--num_timesteps", type=int, default=50) # 500
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--eval_every_n_epochs", type=int, default=1)
argparser.add_argument("--num_samples_per_eval", type=int, default=10)

argparser.add_argument("--accum_steps", type=int, default=1)
argparser.add_argument("--truncate_steps", type=int, default=10)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument("--alpha", type=float, default=0.001)
argparser.add_argument("--gumbel_softmax_temp", type=float, default=0.5)

argparser.add_argument("--decoding", type=str, default='original')
argparser.add_argument("--dps_scale", type=float, default=10)
argparser.add_argument("--tds_alpha", type=float, default=0.5)
argparser.add_argument("--base_model", type=str, default='old')

args = argparser.parse_args()
pdb_path = os.path.join(args.base_path, 'proteindpo_data/AlphaFold_model_PDBs')
max_len = 75  # Define the maximum length of proteins
dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
loader = DataLoader(dataset, batch_size=1000, shuffle=False)

# make a dict of pdb filename: index
for batch in loader:
    pdb_structures = batch[0]
    pdb_filenames = batch[1]
    pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
    break

dpo_dict_path = os.path.join(args.base_path, 'proteindpo_data/processed_data')
dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
old_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
old_fmif_model.to(device)
old_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'])
old_fmif_model.finetune_init()

if args.base_model == 'new':
    new_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    new_fmif_model.to(device)
    new_fmif_model.finetune_init()
    new_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_rewardbp/finetuned_if_model.ckpt'))['model_state_dict'])
    model_to_test_list = [new_fmif_model]
elif args.base_model == 'old':
    model_to_test_list = [old_fmif_model]
elif args.base_model == 'zero_alpha':
    zero_alpha_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    zero_alpha_fmif_model.to(device)
    zero_alpha_fmif_model.finetune_init()
    zero_alpha_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_rewardbp/zeroalpha_if_model.ckpt'))['model_state_dict'])
    model_to_test_list = [zero_alpha_fmif_model]

reward_model = ProteinMPNNOracle(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
reward_model.to(device)
reward_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'])
reward_model.finetune_init()
reward_model.eval()

reward_model_eval = ProteinMPNNOracle(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
reward_model_eval.to(device)
reward_model_eval.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_eval.pt'))['model_state_dict'])
reward_model_eval.finetune_init()
reward_model_eval.eval()

folding_cfg = {
    'seq_per_sample': 1,
    'folding_model': 'esmf',
    'own_device': False,
    'pmpnn_path': './ProteinMPNN/',
    'pt_hub_dir': os.path.join(args.base_path, '.cache/torch/'),
    'colabfold_path': os.path.join(args.base_path, 'colabfold-conda/bin/colabfold_batch') # for AF2
}
folding_cfg = SimpleNamespace(**folding_cfg)
the_folding_model = folding_model.FoldingModel(folding_cfg)
path_for_outputs = os.path.join(args.base_path, 'protein_rewardbp')
save_path = os.path.join(path_for_outputs, 'eval')

noise_interpolant = Interpolant(args)
noise_interpolant.set_device(device)

set_seed(args.seed, use_cuda=True)

for testing_model in model_to_test_list:
    testing_model.eval()
    print(f'Testing Model... Sampling {args.decoding}')
    repeat_num=16
    valid_sp_acc, valid_sp_weights = 0., 0.
    results_merge = []
    all_model_logl = []
    rewards_eval = []
    rewards = []
    for _step, batch in tqdm(enumerate(loader_test)):
        for item_idx in range(8):
            X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
            X = X.repeat(repeat_num, 1, 1, 1)
            mask = mask.repeat(repeat_num, 1)
            chain_M = chain_M.repeat(repeat_num, 1)
            residue_idx = residue_idx.repeat(repeat_num, 1)
            chain_encoding_all = chain_encoding_all.repeat(repeat_num, 1)
            if args.decoding == 'cg':
                S_sp, _, _ = noise_interpolant.sample_controlled_CG(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                    guidance_scale=args.dps_scale, reward_model=reward_model)
            elif args.decoding == 'smc':
                S_sp, _, _ = noise_interpolant.sample_controlled_SMC(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                    reward_model=reward_model, alpha=args.tds_alpha)
            elif args.decoding == 'tds': 
                S_sp, _, _ = noise_interpolant.sample_controlled_TDS(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                    reward_model=reward_model, alpha=args.tds_alpha, guidance_scale=args.dps_scale) 
            elif args.decoding == 'original':
                S_sp, _, _ = noise_interpolant.sample(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all)

            dg_pred = reward_model(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
            rewards.append(dg_pred.detach().cpu().numpy())
            dg_pred_eval = reward_model_eval(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
            rewards_eval.append(dg_pred_eval.detach().cpu().numpy())
            true_false_sp = (S_sp == S).float()
            mask_for_loss = mask*chain_M
            valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, args, item_idx, args.base_path)
            results_merge.extend(results_list)

    valid_sp_accuracy = valid_sp_acc / valid_sp_weights
    print('Sequence recovery accuracy: ', valid_sp_accuracy)

    rewards_eval = np.hstack(rewards_eval)
    rewards = np.hstack(rewards)
    print('Mean reward: ', rewards_eval.mean(), "Positive reward prop %f"%np.mean(rewards_eval>0), "Mean reward (ft): ", rewards.mean(), "Positive reward prop (ft) %f"%np.mean(rewards>0))
    print('median reward: ', np.median(rewards_eval), 'median reward (ft): ', np.median(rewards))

    results_merge = pd.concat(results_merge)
    avg_rmsd = results_merge['gen_true_bb_rmsd'].mean()
    mid_rmsd = results_merge['gen_true_bb_rmsd'].median()
    rmsd_rate = results_merge['gen_true_bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
    print('Median gen_true RMSD: ', mid_rmsd, 'Mean gen_true RMSD: ', avg_rmsd, 'Good RMSD prop: ', rmsd_rate)

    results_merge['rewards_eval'] = rewards_eval
    results_merge['rewards'] = rewards
    results_merge['success'] = (results_merge['gen_true_bb_rmsd'] < 2) & (results_merge['rewards_eval'] > 0)

    success_rate = results_merge['success'].mean()
    print('success rate: ', success_rate)

    results_merge.to_csv(f'./eval_results/{args.decoding}_{args.base_model}_{args.dps_scale}_{args.tds_alpha}_{args.seed}_results_merge.csv')

    results_dict = {'Sequence Recovery Accuracy': valid_sp_accuracy, 
                    'Model Log Likelihood': all_model_logl.mean(), 
                    'Mean Reward': rewards_eval.mean(), 
                    'Positive Reward Proportion': np.mean(rewards_eval>0), 
                    'Mean Reward (ft)': rewards.mean(), 
                    'Positive Reward Proportion (ft)': np.mean(rewards>0), 
                    'Median Reward': np.median(rewards_eval), 
                    'Median Reward (ft)': np.median(rewards), 
                    'Median gen_true RMSD': mid_rmsd, 
                    'Mean gen_true RMSD': avg_rmsd,
                    'Good RMSD Proportion': rmsd_rate,
                    'Success Rate': success_rate} 
    results_df_final = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
    results_df_final.to_csv(f'./eval_results/{args.decoding}_{args.base_model}_{args.dps_scale}_{args.tds_alpha}_{args.seed}_results_summary.csv')
