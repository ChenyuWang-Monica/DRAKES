import torch
import grelu
import pandas as pd
import os
from grelu.lightning import LightningModel
import grelu.data.preprocess
import grelu.data.dataset
import dataloader_gosai
import numpy as np
from typing import Callable, Union, List
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
import torch.nn.functional as F

base_path = '/data/scratch/wangchy/seqft/'


def get_gosai_oracle(mode='train'):
    if mode == 'train':
        model_load = LightningModel.load_from_checkpoint(os.path.join(base_path, 'mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt'), map_location='cuda')
    elif mode == 'eval':
        model_load = LightningModel.load_from_checkpoint(os.path.join(base_path, 'mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt'), map_location='cuda')
    else:
        raise ValueError
    model_load.train_params['logger'] = None
    return model_load

def cal_gosai_pred(seqs, model=None, mode='eval'):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = get_gosai_oracle(mode=mode)
    df_seqs = pd.DataFrame(seqs, columns=['seq'])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    preds = model.predict_on_dataset(pred_dataset, devices=[0])
    return preds.squeeze() # numpy array with shape [n_seqs, 3]

def cal_gosai_pred_new(seqs, model=None, mode='eval'):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = get_gosai_oracle(mode=mode)
    model.eval()
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long().cuda()
    onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return preds.squeeze()

def cal_atac_pred(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = LightningModel.load_from_checkpoint(os.path.join(base_path, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt'), map_location='cuda')
    df_seqs = pd.DataFrame(seqs, columns=['seq'])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    preds = model.predict_on_dataset(pred_dataset, devices=[0])
    return preds.squeeze() # numpy array with shape [n_seqs, 7]


def cal_atac_pred_new(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = LightningModel.load_from_checkpoint(os.path.join(base_path, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt'), map_location='cuda')
    model.eval()
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long().cuda()
    onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return preds.squeeze() # numpy array with shape [n_seqs, 7]


def count_kmers(seqs, k=3):
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i : i + k]
            try:
                counts[subseq] += 1
            except KeyError:
                counts[subseq] = 1
    return counts


def subset_for_eval(n=5000, seed=0):
    train_set = dataloader_gosai.get_datasets_gosai()
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set_sp = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), n, replace=False))
    return train_set_sp 


def subset_eval_groundtruth(sets_sp):
    train_set_sp = sets_sp
    train_set_sp_clss = train_set_sp.dataset.clss[train_set_sp.indices]
    return train_set_sp_clss 


def subset_eval_preds(sets_sp, oracle_model=None):
    train_set_sp = sets_sp
    train_preds = cal_gosai_pred(
        dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy()), oracle_model)
    return train_preds


def subset_eval_kmers(sets_sp, k=3):
    train_set_sp = sets_sp
    train_seqs = dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy())
    train_kmers = count_kmers(train_seqs, k)
    return train_kmers


def subset_eval_embs(sets_sp, oracle_model=None):
    train_set_sp = sets_sp
    train_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy()), oracle_model)
    return train_sp_emb 


def cal_emb_pca(sets_sp, n_components=50, oracle_model=None):
    train_set_sp = sets_sp
    train_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(train_set_sp.dataset.seqs[train_set_sp.indices].numpy()), oracle_model)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(train_sp_emb.reshape(train_sp_emb.shape[0], -1))
    return pca


def subset_eval_embs_pca(sets_sp, pca, oracle_model=None):
    train_sp_emb = subset_eval_embs(sets_sp, oracle_model)
    train_sp_emb_pca = pca.transform(train_sp_emb.reshape(train_sp_emb.shape[0], -1))
    return train_sp_emb_pca 


# https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py
def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist


def embed_on_dataset(
    model,
    dataset: Callable,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 256,
):
    """
    Return embeddings for a dataset of sequences

    Args:
        dataset: Dataset object that yields one-hot encoded sequences
        devices: Device IDs to use
        num_workers: Number of workers for data loader
        batch_size: Batch size for data loader

    Returns:
        Numpy array of shape (B, T, L) containing embeddings.
    """
    torch.set_float32_matmul_precision("medium")

    # Make dataloader
    dataloader = model.make_predict_loader(
        dataset, num_workers=num_workers, batch_size=batch_size
    )

    # Get device
    orig_device = model.device
    device = model.parse_devices(devices)[1]
    if isinstance(device, list):
        device = device[0]
    model.to(device)

    # Get embeddings
    preds = []
    model.model = model.model.eval()
    for batch in iter(dataloader):
        batch = batch.to(device)
        preds.append(model.model.embedding(batch).detach().cpu())

    # Return to original device
    model.to(orig_device)
    return torch.vstack(preds).numpy()


def cal_gosai_emb(seqs, model=None):
    """
    seqs: list of sequences (detokenized ACGT...)
    """
    if model is None:
        model = get_gosai_oracle()
    df_seqs = pd.DataFrame(seqs, columns=['seq'])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    embs = embed_on_dataset(model, pred_dataset, devices=[0])
    return embs # numpy array with shape [n_seqs, 3072, 2]


def cal_highexp_kmers(k=3, return_clss=False):
    train_set = dataloader_gosai.get_datasets_gosai()
    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.99) # 4.56
    highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
    highexp_kmers_99 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_99 = len(highexp_indices)

    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.999) # 6.27
    highexp_indices = [i for i, data in enumerate(train_set) if data['clss'][0] > exp_threshold]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy())
    highexp_kmers_999 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_999 = len(highexp_indices)

    if return_clss:
        highexp_set_sp_clss_999 = highexp_set_sp.dataset.clss[highexp_set_sp.indices]
        highexp_preds_999 = cal_gosai_pred_new(
                dataloader_gosai.batch_dna_detokenize(highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy()))
        return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs
    
    return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999


def cal_kmer_corr(model, highexp_kmers, n_highexp_kmers, n_sample=128):
    model.eval()
    all_detoeknized_samples = []
    for _ in range(10):
        samples = model._sample(eval_sp_size=n_sample).detach().cpu().numpy()
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples)
        all_detoeknized_samples.extend(detokenized_samples)
    generated_kmer = count_kmers(all_detoeknized_samples)


    kmer_set = set(highexp_kmers.keys()) | set(generated_kmer.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in highexp_kmers:
            counts[i][1] = highexp_kmers[kmer] * len(generated_kmer) / n_highexp_kmers
        if kmer in generated_kmer:
            counts[i][0] = generated_kmer[kmer]
    
    corr = pearsonr(counts[:, 0], counts[:, 1])[0]
    return corr

def cal_avg_likelihood(model, old_model, n_sample=128):
    model.eval()
    old_model.eval()
    all_raw_samples = []
    for _ in range(10):
        samples = model._sample(eval_sp_size=n_sample)
        all_raw_samples.append(samples)
    all_raw_samples = torch.concat(all_raw_samples)
    avg_likelihood = old_model._forward_pass_diffusion(all_raw_samples).sum(-1).mean().item()
    return avg_likelihood
