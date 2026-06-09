"""PriFold-SymFlow: PriFold data/MARS + SymFold-style DiT flow matching.

Versions:
    v1 - Baseline: 6-layer Axial DiT + outer-concat MARS hidden + pos_bias channel.
         Train: 53/60 epoch on combined bpRNA TR0 + RNAStrAlign tr.
         Best val F1=0.5793, bpRNA-test=0.258, rnastralign-test=0.753, archiveii=0.510.
    v2 - DA-SE-DiT-MARS: 9-layer Dilated Axial + Triangle Update + SwiGLU + RoPE
         + MARS multi-layer attention map + density conditioning.
         Two independent models trained per dataset (bpRNA / RNAStrAlign).
"""
