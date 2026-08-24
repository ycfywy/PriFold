import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from collections import OrderedDict
from tqdm import tqdm

from utils.lm import get_extractor
from utils.tools import get_posbias
from utils.RNAformer.model.Riboformer_outfirst import RiboFormer
from utils.configuration import Config
from utils.predictor import SSDataset

import pandas as pd

MODEL_PATH_MAP = {
    'bprna': 'ss_model_bprna.pth',
    'rnastralign': 'ss_model_rnastralign.pth',
    'archiveii': 'ss_model_rnastralign.pth',
}


def collate_fn(batch, scale):
    seqs, cts, _ = zip(*batch)
    max_len = max([len(seq) + 2 for seq in seqs])
    data_dict = tokenizer.batch_encode_plus(seqs, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    data_dict['pos_bias'] = get_posbias(seqs, max_len, scale)

    ct_masks = [np.ones(ct.shape) for ct in cts]
    cts = [np.pad(ct, (0, max_len - ct.shape[0]), 'constant') for ct in cts]
    ct_masks = [np.pad(ct_mask, (0, max_len - ct_mask.shape[0]), 'constant') for ct_mask in ct_masks]
    data_dict['ct'] = torch.FloatTensor(cts)
    data_dict['ct_mask'] = torch.FloatTensor(ct_masks)
    data_dict['seq_len'] = torch.tensor([len(seq) for seq in seqs])
    return data_dict


def build_dataset(mode, data_dir, max_len, tokenizer):
    if mode == 'bprna':
        df = pd.read_csv(f'{data_dir}/bprna/bpRNA.csv')
        df = df[df['seq'].str.len() < max_len]
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        test_dataset = SSDataset(df_test, data_path=f'{data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=None, smooth=None)
    elif mode == 'rnastralign':
        df = pd.read_csv(f'{data_dir}/RNAStrAlign/rnastralign.csv')
        df = df[df['seq'].str.len() < max_len]
        df_test = df[df['data_name'] == 'ts'].reset_index(drop=True)
        test_dataset = SSDataset(df_test, data_path=f'{data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=None, smooth=None)
    elif mode == 'archiveii':
        df = pd.read_csv(f'{data_dir}/archiveII/archiveII.csv')
        df = df[df['seq'].str.len() < max_len]
        test_dataset = SSDataset(df, data_path=f'{data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)
    else:
        raise ValueError(f'unknown mode: {mode}')
    return test_dataset


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = Config(config_file=args.config)
    model = RiboFormer(model_config['RNAformer'], extractor, args.is_freeze)

    checkpoint = torch.load(args.model_path)
    original_state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for key, value in original_state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    device = torch.device('cuda')
    model.eval()
    model.to(device)

    test_dataset = build_dataset(args.mode, args.data_dir, args.max_len, tokenizer)
    print(f"[{args.mode} <{args.max_len}] len of dataset: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=partial(collate_fn, scale=args.scale))

    # Cache all per-sample probabilities and labels (model forward only once).
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data_dict in tqdm(test_loader, desc=f'{args.mode} <{args.max_len} forward'):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(device)
            logits = model(data_dict)
            labels = data_dict['ct']
            bs = logits.shape[0]
            for idx in range(bs):
                seq_length = data_dict['attention_mask'][idx].sum().item()
                logit = logits[idx, :seq_length, :seq_length]
                label = labels[idx, :logit.shape[0], :logit.shape[1]]
                probs = torch.sigmoid(logit).detach().cpu().numpy()
                all_probs.append(probs.reshape(-1))
                all_labels.append(label.detach().cpu().numpy().reshape(-1))
            del logits, labels, data_dict
            torch.cuda.empty_cache()

    # Precompute per-sample positive counts and flatten for vectorized eval.
    n_samples = len(all_labels)
    pos_counts = np.array([lab.sum() for lab in all_labels])  # number of positive (paired) positions per sample

    thresholds = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
    best_thr, best_f1 = None, -1.0
    print(f"\n{'threshold':>10} {'precision':>12} {'recall':>12} {'f1':>12}")
    for thr in thresholds:
        prec_sum, rec_sum, f1_sum = 0.0, 0.0, 0.0
        for i in range(n_samples):
            pred = all_probs[i] > thr
            tp = int(np.logical_and(pred, all_labels[i]).sum())
            fp = int(pred.sum()) - tp
            fn = int(pos_counts[i]) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            prec_sum += prec
            rec_sum += rec
            f1_sum += f1
        prec = prec_sum / n_samples
        rec = rec_sum / n_samples
        f1 = f1_sum / n_samples
        print(f'{thr:>10.2f} {prec:>12.6f} {rec:>12.6f} {f1:>12.6f}')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec, best_rec = prec, rec

    print(f"\n=== [{args.mode} <{args.max_len}] Best F1 ===")
    print(f"threshold: {best_thr:.2f}, precision: {best_prec:.6f}, recall: {best_rec:.6f}, F1: {best_f1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_scale', type=str, default='160m')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument('--pretrained_lm_dir', type=str, default='./model')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--config', type=str, default='./utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml')
    parser.add_argument('--scale', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--max_len', type=int, default=490)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join(args.pretrained_lm_dir, MODEL_PATH_MAP[args.mode])

    extractor, tokenizer = get_extractor(args)
    main(args)
