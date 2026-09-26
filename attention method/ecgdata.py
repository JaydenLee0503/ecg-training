"""Prepare/verify/dry-run ECGData attention; training is a separate explicit action."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from attention_ecg.ecgdata import (PROTOCOL, REPO, DEFAULT_CACHE, prepare_cache,
                                  verify_cache, read_json, fit_normalizer, normalize)


def dry_run(cache):
    import numpy as np
    import torch
    from attention_ecg.model import CepstralSwin
    from attention_ecg.ecgdata_training import partition

    manifest,data=verify_cache(cache)
    protocol=manifest['protocol']
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model=CepstralSwin(**protocol['model']).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    # Only inspect fit patients: outer fold 0 contributes no signals here.
    tr,_=partition(data,0)
    mean,scale=fit_normalizer(data['X'],tr)
    inputs=normalize(data['X'][tr[:2]],mean,scale)
    with torch.inference_mode():
        logits=model(torch.from_numpy(inputs))
    if not torch.isfinite(logits).all() or not all(torch.equal(before[k],v) for k,v in model.state_dict().items()):
        raise ValueError('Forward-only check failed')
    return dict(status='forward_only',device='cpu',training_started=False,weights_unchanged=True,
                cache_shape=list(data['X'].shape),parameters=sum(p.numel() for p in model.parameters()),
                logits_shape=list(logits.shape),patient_count=len(np.unique(data['patients'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('prepare','verify','dry-run','train','report'):
        p=sub.add_parser(action)
        p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
        if action=='prepare':
            p.add_argument('--mat',type=Path,default=REPO/'ECGData.mat')
        if action in ('train','report'):
            p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results/ecgdata_lfcc_swin_v1')
        if action=='train':
            p.add_argument('--execute-training',action='store_true',help='Explicitly enable model fitting')
            p.add_argument('--device',choices=('cpu','cuda'),required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare_cache(args.mat,args.cache)
    elif args.action=='verify':
        _,data=verify_cache(args.cache)
        result=dict(status='verified',shape=list(data['X'].shape),training_started=False)
    elif args.action=='dry-run':
        result=dry_run(args.cache)
    elif args.action=='train':
        from attention_ecg.ecgdata_training import run_training
        result=run_training(args.cache,args.out,execute=args.execute_training,device=args.device)
    else:
        from attention_ecg.ecgdata_report import make_report
        result=make_report(args.cache,args.out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
