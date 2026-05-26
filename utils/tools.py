import torch
import numpy as np
from utils.predictor import SSDataset
import pandas as pd

def load_data(args, tokenizer, aug=None, smooth=None):
    if args.mode == 'bprna':
        ## bprna data ##
        df = pd.read_csv(f'{args.data_dir}/bprna/bpRNA.csv')

        df = df[df['seq'].str.len() < 490]
        df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
        df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/bprna/ct/TR0', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/bprna/ct/VL0', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=None, smooth=None)
    
    elif args.mode == 'bprna-test':
        ## bprna data ##
        df = pd.read_csv(f'{args.data_dir}/bprna/bpRNA.csv')
        df = df[df['seq'].str.len() < 490]
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        train_dataset = None
        val_dataset = None
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/bprna/ct/TS0', tokenizer=tokenizer, aug=None, smooth=None)

    elif args.mode == 'archiveii-test':
        df = pd.read_csv(f'{args.data_dir}/archiveII/archiveII.csv')
        df = df[df['seq'].str.len() < 490]
        train_dataset = None
        val_dataset = None
        test_dataset = SSDataset(df, data_path=f'{args.data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)
    
    elif args.mode == 'rnastralign-test':
        df_str = pd.read_csv(f'{args.data_dir}/RNAStrAlign/rnastralign.csv')
        df_str = df_str[df_str['seq'].str.len() < 490]
        df_val = df_str[df_str['data_name'] == 'ts'].reset_index(drop=True)
        train_dataset = None
        val_dataset = None
        test_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=None, smooth=None)

    elif args.mode == 'rnastralign':
        df_str = pd.read_csv(f'{args.data_dir}/RNAStrAlign/rnastralign.csv')
        df_arc = pd.read_csv(f'{args.data_dir}/archiveII/archiveII.csv')

        df_str = df_str[df_str['seq'].str.len() < 490]
        df_arc = df_arc[df_arc['seq'].str.len() < 490]

        df_train = df_str[df_str['data_name'] == 'tr'].reset_index(drop=True)
        df_val = df_str[df_str['data_name'] == 'ts'].reset_index(drop=True)
        df_test = df_arc.reset_index(drop=True)

        train_dataset = SSDataset(df_train, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=aug, smooth=smooth)
        val_dataset = SSDataset(df_val, data_path=f'{args.data_dir}/RNAStrAlign', tokenizer=tokenizer, aug=None, smooth=None)
        test_dataset = SSDataset(df_test, data_path=f'{args.data_dir}/archiveII/ct', tokenizer=tokenizer, aug=None, smooth=None)

    return train_dataset, val_dataset, test_dataset

def get_posbias(seqs, max_len, scale):
    pair_scores = {'AT': 3, 'TA': 3, 'GC': 6, 'CG': 6, 'GT': 1, 'TG': 1}
    posbias = np.ones((len(seqs), max_len-2, max_len-2), dtype=np.float32)
    for i, seq in enumerate(seqs):
        L = len(seq)
        seq_arr = np.array(list(seq))
        for pair, score in pair_scores.items():
            mask_row = (seq_arr == pair[0])
            mask_col = (seq_arr == pair[1])
            posbias[i, :L, :L] += np.outer(mask_row, mask_col) * (score * scale)
    posbias = np.pad(posbias, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    return torch.Tensor(posbias)