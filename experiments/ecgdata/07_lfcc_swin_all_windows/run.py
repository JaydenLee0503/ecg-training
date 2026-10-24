"""ECG-07 command line: prepare | verify | train --execute-training --device cuda | report."""
import argparse
import json

import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('prepare', 'verify', 'train', 'report'):
        p = sub.add_parser(action)
        if action == 'train':
            p.add_argument('--execute-training', action='store_true', help='Explicitly enable model fitting')
            p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = study.prepare()
    elif args.action == 'verify':
        data = study.verify_cache()
        result = dict(status='verified', shape=list(data['X'].shape), training_started=False)
    elif args.action == 'train':
        result = study.train(args.execute_training, args.device)
    else:
        result = study.report()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
