import argparse
import random
import string
import datetime
from datetime import date

runid = ''.join(random.choice(string.ascii_letters) for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

def main(args):
    import time, os
    import numpy as np
    import torch
    import queue
    import os.path
    from concurrent.futures import ProcessPoolExecutor    
    from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
    from fmif.model_utils import featurize, loss_smoothed, loss_nll, ProteinMPNNFMIF
    from fmif.fm_utils import Interpolant, fm_model_step
    from tqdm import tqdm
    import wandb

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    path_for_outputs = os.path.join(args.base_path, 'pmpnn/outputs') if not args.debug else os.path.join(args.base_path, 'pmpnn/outputs_debug')
    base_folder = time.strftime(path_for_outputs, time.localtime())

    base_folder = os.path.join(base_folder, runid)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if base_folder[-1] != '/':
        base_folder += '/'
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    
    assert torch.cuda.is_available(), "CUDA is not available"
    # set random seed
    set_seed(args.seed, use_cuda=True)

    # wandb
    if not args.debug:
        wandb.init(project='pmpnn', name=runid, dir=base_folder, config=args)
        curr_date = date.today().strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
    else:
        with open(logfile, 'a') as f:
            f.write("Debug mode, not logging to wandb\n")

    with open (logfile, 'a') as f:
        f.write(f"Run ID: {runid}\n")
        f.write(f"Arguments: {args}\n")

    data_path = os.path.join(args.base_path, 'pmpnn/raw/pdb_2021aug02')
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    print(len(train), len(valid), len(test)) # 23349 1464 1539

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = ProteinMPNNFMIF(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)

    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)


    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        pq = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            pq.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
        pdb_dict_test = pq.get().result()
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
        print(len(loader_train), len(loader_valid), len(loader_test))
        
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    pdb_dict_test = pq.get().result()
                    dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
                    loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    pq.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = fm_model_step(model, noisy_batch)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_av_smoothed).backward()
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = fm_model_step(model, noisy_batch)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            validation_sp_accuracy_ = '-'
            test_sp_accuracy_ = '-'

            if (e+1) % args.eval_every_n_epochs == 0:
                with torch.no_grad():
                    print(len(loader_valid))
                    valid_sp_acc, valid_sp_weights = 0., 0.
                    for _, batch in tqdm(enumerate(loader_valid)):
                        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                        S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
                        true_false_sp = (S_sp == S).float()
                        mask_for_loss = mask*chain_M
                        valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                        valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    validation_sp_accuracy = valid_sp_acc / valid_sp_weights
                    validation_sp_accuracy_ = np.format_float_positional(np.float32(validation_sp_accuracy), unique=False, precision=3)

                    print(len(loader_test))
                    test_sp_acc, test_sp_weights = 0., 0.
                    for _, batch in tqdm(enumerate(loader_test)):
                        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                        S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
                        true_false_sp = (S_sp == S).float()
                        mask_for_loss = mask*chain_M
                        test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                        test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    test_sp_accuracy = test_sp_acc / test_sp_weights
                    test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False, precision=3)
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}')
            
            if not args.debug:
                wandb.log({"train_perplexity": train_perplexity, "valid_perplexity": validation_perplexity, "train_accuracy": train_accuracy, "valid_accuracy": validation_accuracy}, step=total_step, commit=False)
                if (e+1) % args.eval_every_n_epochs== 0:
                    wandb.log({"valid_sp_accuracy": validation_sp_accuracy, "test_sp_accuracy": test_sp_accuracy}, step=total_step)
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--base_path", type=str, default="/data/scratch/wangchy/seqft/", help="base path for data and model") 
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=400, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0)
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--do_purity", type=bool, default=True)
    argparser.add_argument("--num_timesteps", type=int, default=500)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--eval_every_n_epochs", type=int, default=20)
 
    args = argparser.parse_args()    
    main(args)
