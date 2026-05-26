from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.predictor import Augmentation


@dataclass
class RNARecord:
    dataset: str
    split: str
    file_name: str
    seq: str
    ct_path: str


def _filtered(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    return df[df["seq"].astype(str).str.len() < max_len].copy()


def build_records(data_dir: str, stage: str, max_len: int = 490) -> list[RNARecord]:
    records: list[RNARecord] = []

    def add_bprna(split_name: str, data_name: str):
        df = pd.read_csv(os.path.join(data_dir, "bprna", "bpRNA.csv"))
        df = _filtered(df, max_len)
        df = df[df["data_name"] == data_name]
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            records.append(RNARecord(
                dataset="bpRNA",
                split=split_name,
                file_name=file_name,
                seq=str(row["seq"]),
                ct_path=os.path.join(data_dir, "bprna", "ct", data_name, f"{file_name}.npy"),
            ))

    def add_rnastr(split_name: str, data_name: str):
        df = pd.read_csv(os.path.join(data_dir, "RNAStrAlign", "rnastralign.csv"))
        df = _filtered(df, max_len)
        df = df[df["data_name"] == data_name]
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            records.append(RNARecord(
                dataset="RNAStrAlign",
                split=split_name,
                file_name=file_name,
                seq=str(row["seq"]),
                ct_path=os.path.join(data_dir, "RNAStrAlign", f"{file_name}.npy"),
            ))

    def add_archive():
        df = pd.read_csv(os.path.join(data_dir, "archiveII", "archiveII.csv"))
        df = _filtered(df, max_len)
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            records.append(RNARecord(
                dataset="ArchiveII",
                split="test",
                file_name=file_name,
                seq=str(row["seq"]),
                ct_path=os.path.join(data_dir, "archiveII", "ct", f"{file_name}.npy"),
            ))

    if stage == "train":
        add_bprna("train", "TR0")
        add_rnastr("train", "tr")
    elif stage == "val":
        add_bprna("val", "VL0")
        add_rnastr("val/test", "ts")
    elif stage == "bprna-test":
        add_bprna("test", "TS0")
    elif stage == "rnastralign-test":
        add_rnastr("val/test", "ts")
    elif stage == "archiveii-test":
        add_archive()
    elif stage == "bprna-val":
        add_bprna("val", "VL0")
    elif stage == "rnastralign-vl":
        add_rnastr("unused(vl)", "vl")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return records


def _one_hot(seq: str, length: int) -> np.ndarray:
    out = np.zeros((length, 4), dtype=np.float32)
    mapping = {"A": 0, "T": 1, "G": 2, "C": 3}
    for idx, base in enumerate(seq[:length]):
        j = mapping.get(base)
        if j is not None:
            out[idx, j] = 1.0
    return out


def _pos_bias(seq: str, length: int, scale: float) -> np.ndarray:
    pair_scores = {"AT": 3, "TA": 3, "GC": 6, "CG": 6, "GT": 1, "TG": 1}
    out = np.ones((length, length), dtype=np.float32)
    arr = np.array(list(seq))
    raw_len = len(seq)
    for pair, score in pair_scores.items():
        row = arr == pair[0]
        col = arr == pair[1]
        out[:raw_len, :raw_len] += np.outer(row, col) * (score * scale)
    return out


class PriFoldSymFlowDataset(Dataset):
    def __init__(
        self,
        records: list[RNARecord],
        augment: bool = False,
        select: float = 0.1,
        replace: float = 0.3,
        limit: int | None = None,
    ):
        if limit is not None and limit > 0:
            records = records[:limit]
        self.records = records
        self.aug = Augmentation(select, replace) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        seq = rec.seq.upper().replace("U", "T")
        ct = np.load(rec.ct_path).astype(np.float32)
        if self.aug is not None:
            seq = self.aug(seq, ct)
        return {
            "seq": seq,
            "contact": ct,
            "length": len(seq),
            "name": rec.file_name,
            "dataset": rec.dataset,
            "split": rec.split,
        }


class LengthBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, lengths: Iterable[int], batch_size: int, shuffle: bool = True, seed: int = 0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        order.sort(key=lambda i: self.lengths[i])
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def make_collate_fn(tokenizer, patch_size: int = 4, pos_bias_scale: float = 0.01):
    def collate(batch):
        seqs = [item["seq"] for item in batch]
        lengths = np.array([item["length"] for item in batch], dtype=np.int64)
        max_l = int(lengths.max())
        set_len = int(math.ceil(max_l / patch_size) * patch_size)
        contacts, masks, seq_oh, pos_bias = [], [], [], []
        for item in batch:
            length = item["length"]
            ct = np.zeros((set_len, set_len), dtype=np.float32)
            raw_ct = item["contact"][:length, :length]
            ct[: raw_ct.shape[0], : raw_ct.shape[1]] = raw_ct
            mask = np.zeros((set_len, set_len), dtype=np.float32)
            mask[:length, :length] = 1.0
            contacts.append(ct)
            masks.append(mask)
            oh = np.zeros((set_len, 4), dtype=np.float32)
            oh[:length] = _one_hot(item["seq"], length)
            seq_oh.append(oh)
            pb = np.zeros((set_len, set_len), dtype=np.float32)
            pb[:length, :length] = _pos_bias(item["seq"], length, pos_bias_scale)[:length, :length]
            pos_bias.append(pb)

        tokenized = tokenizer.batch_encode_plus(
            seqs,
            padding="max_length",
            max_length=max_l + 2,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "seqs": seqs,
            "seq_oh": torch.from_numpy(np.stack(seq_oh)).float(),
            "contact": torch.from_numpy(np.stack(contacts)).unsqueeze(1).float(),
            "contact_mask": torch.from_numpy(np.stack(masks)).unsqueeze(1).float(),
            "pos_bias": torch.from_numpy(np.stack(pos_bias)).float(),
            "length": torch.from_numpy(lengths).long(),
            "set_max_len": set_len,
            "names": [item["name"] for item in batch],
            "datasets": [item["dataset"] for item in batch],
        }
    return collate


def build_loader(stage: str, config: dict, tokenizer, shuffle: bool):
    tcfg = config["training"]
    records = build_records(config["paths"]["data_dir"], stage, max_len=tcfg.get("max_len_filter", 490))
    limit_key = f"max_{stage.replace('-', '_')}_samples"
    limit = tcfg.get(limit_key)
    if stage == "train":
        limit = tcfg.get("max_train_samples", limit)
    elif stage == "val":
        limit = tcfg.get("max_val_samples", limit)
    dataset = PriFoldSymFlowDataset(
        records,
        augment=(stage == "train" and tcfg.get("augmentation", {}).get("enabled", False)),
        select=tcfg.get("augmentation", {}).get("select", 0.1),
        replace=tcfg.get("augmentation", {}).get("replace", 0.3),
        limit=limit,
    )
    sampler = LengthBucketBatchSampler(
        [len(r.seq) for r in dataset.records],
        batch_size=tcfg.get("batch_size", 1),
        shuffle=shuffle,
        seed=config.get("seed", 3407),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(tokenizer, config["model"].get("patch_size", 4), config["model"].get("pos_bias_scale", 0.01)),
        num_workers=tcfg.get("num_workers", 0),
        pin_memory=tcfg.get("pin_memory", False),
    )
    return loader
