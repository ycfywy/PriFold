import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.lm import get_extractor
from utils.tools import get_posbias
from utils.RNAformer.model.Riboformer_outfirst import RiboFormer
from utils.configuration import Config


class BpRNANewDataset(Dataset):
    def __init__(self, records):
        self.records = records
        print(f"len of dataset: {len(records)}")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def structure_to_contact_map(structure):
        contact_map = np.zeros((len(structure), len(structure)), dtype=np.float32)
        stack = []
        for index, char in enumerate(structure):
            if char == '(':
                stack.append(index)
            elif char == ')' and stack:
                left = stack.pop()
                contact_map[left, index] = 1.0
                contact_map[index, left] = 1.0
        return contact_map

    def __getitem__(self, index):
        row = self.records[index]
        sequence = str(row['sequence']).upper().replace('U', 'T')
        structure = str(row['secondary_structure'])
        if len(sequence) != len(structure):
            raise ValueError(f"sequence/structure length mismatch for {row.get('id', index)}")
        contact_map = self.structure_to_contact_map(structure)
        return sequence, contact_map, row.get('id', index)


def collate_fn(batch, scale):
    seqs, cts, _ = zip(*batch)
    max_len = max(len(seq) + 2 for seq in seqs)
    data_dict = tokenizer.batch_encode_plus(
        seqs, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt'
    )
    data_dict['pos_bias'] = get_posbias(seqs, max_len, scale)

    cts = [np.pad(ct, (0, max_len - ct.shape[0]), 'constant') for ct in cts]
    ct_masks = [np.pad(np.ones(ct.shape), (0, max_len - ct.shape[0]), 'constant') for ct in cts]
    data_dict['ct'] = torch.FloatTensor(cts)
    data_dict['ct_mask'] = torch.FloatTensor(ct_masks)
    data_dict['seq_len'] = torch.tensor([len(seq) for seq in seqs])
    return data_dict


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = Config(config_file=args.config)
    model = RiboFormer(model_config['RNAformer'], extractor, args.is_freeze)
    checkpoint = torch.load(args.model_path)
    original_state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for key, value in original_state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    device = torch.device('cuda')
    model.to(device).eval()
    dataset = BpRNANewDataset(args.records)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=partial(collate_fn, scale=args.scale)
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for data_dict in tqdm(loader, desc='bpRNA-new forward'):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(device)
            logits = model(data_dict)
            labels = data_dict['ct']
            for index in range(logits.shape[0]):
                seq_length = data_dict['attention_mask'][index].sum().item()
                logit = logits[index, :seq_length, :seq_length]
                label = labels[index, :logit.shape[0], :logit.shape[1]]
                all_probs.append(torch.sigmoid(logit).cpu().numpy().reshape(-1))
                all_labels.append(label.cpu().numpy().reshape(-1))
            del logits, labels, data_dict
            torch.cuda.empty_cache()

    n_samples = len(all_labels)
    positive_counts = np.array([label.sum() for label in all_labels])
    thresholds = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
    best = None
    print(f"\n{'threshold':>10} {'precision':>12} {'recall':>12} {'f1':>12}")
    for threshold in thresholds:
        precision_sum = recall_sum = f1_sum = 0.0
        for probs, label, positive_count in zip(all_probs, all_labels, positive_counts):
            pred = probs > threshold
            true_positive = int(np.logical_and(pred, label).sum())
            false_positive = int(pred.sum()) - true_positive
            false_negative = int(positive_count) - true_positive
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
        result = {
            'threshold': float(threshold),
            'precision': precision_sum / n_samples,
            'recall': recall_sum / n_samples,
            'f1': f1_sum / n_samples,
        }
        print(f"{threshold:>10.2f} {result['precision']:>12.6f} {result['recall']:>12.6f} {result['f1']:>12.6f}")
        if best is None or result['f1'] > best['f1']:
            best = result

    print("\n=== [bpRNA-new] Best F1 ===")
    print(
        f"threshold: {best['threshold']:.2f}, precision: {best['precision']:.6f}, "
        f"recall: {best['recall']:.6f}, F1: {best['f1']:.6f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_scale', type=str, default='160m')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--pretrained_lm_dir', type=str, default='./model')
    parser.add_argument('--model_path', type=str, default='./model/ss_model_bprna.pth')
    parser.add_argument('--config', type=str, default='./utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml')
    parser.add_argument('--scale', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    args.records = [json.loads(line) for line in __import__('sys').stdin if line.strip()]
    extractor, tokenizer = get_extractor(args)
    main(args)
