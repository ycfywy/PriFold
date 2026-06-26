# -*- coding: utf-8 -*-
"""不重训 decoder 验证: v9 best.pt 上对比 topk vs matching decoder。"""
import json, sys
from pathlib import Path
import torch
ROOT = Path('/root/aigame/dannyyan/PriFold')
sys.path.insert(0, str(ROOT))
from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.eval.eval_v9 import build_model
from torch.utils.data import DataLoader

cfg = json.load(open(ROOT/'symfold/config/v9/v9_ddp.json'))
dev = torch.device('cuda:0')

class A: pass
a = A(); a.pretrained_lm_dir = str(ROOT/'model'); a.model_scale = 'lx'
extractor, tok = get_extractor(a)
model = build_model(cfg, extractor).to(dev)
ck = torch.load(ROOT/'symfold/outputs/v9_ddp/model/best.pt', map_location=dev, weights_only=False)
model.load_state_dict(ck['model'], strict=False)
model.eval()

recs = build_records(str(ROOT/'data'), 'bprna-test', max_len=490)
ds = PriFoldSymFlowDataset(recs, augment=False)
ld = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate_fn(tok), num_workers=4)
scfg = cfg['sampling']

def run(mode):
    tot = {'f1':0,'precision':0,'recall':0,'mcc':0,'n':0}
    with torch.no_grad():
        for b in ld:
            b = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred,_ = model.predict(b, budget_fraction=scfg['default_budget_fraction'],
                    use_density_budget=scfg['use_density_budget'], score_threshold=scfg['score_threshold'],
                    length_decay=scfg['length_decay'], budget_floor=scfg['budget_floor'], decode_mode=mode)
            m = contact_metrics(pred, b['contact'], b['length'])
            bs = pred.shape[0]
            for k in ['f1','precision','recall','mcc']: tot[k]+=m[k]*bs
            tot['n']+=bs
    n=tot['n']
    return {k:tot[k]/n for k in ['f1','precision','recall','mcc']}, n

r_topk,n = run('topk')
r_match,_ = run('matching')
out = {'n':n,'topk':r_topk,'matching':r_match,
       'delta_f1':r_match['f1']-r_topk['f1']}
Path('/tmp/decoder_val.json').write_text(json.dumps(out,indent=2))
print('DONE', json.dumps(out,indent=2))
