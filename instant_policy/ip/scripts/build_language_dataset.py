import argparse
import os
import torch
import numpy as np
from ip.utils.language_utils import get_language_description, encode_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with data_*.pt files')
    parser.add_argument('--task_name', type=str, required=True, help='RLBench task name')
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help='Sentence-BERT model name')
    parser.add_argument('--device', type=str, default='cpu', help='Device for encoding')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for template selection')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing lang_emb')
    parser.add_argument('--add_text', action='store_true', help='Also store lang_text string')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.data_dir) if f.endswith('.pt')]
    files.sort()

    rng = np.random.default_rng(args.seed)
    descriptions = []
    targets = []

    for fname in files:
        path = os.path.join(args.data_dir, fname)
        data = torch.load(path)
        if hasattr(data, 'lang_emb') and not args.overwrite:
            continue
        desc = get_language_description(args.task_name, rng=rng)
        descriptions.append(desc)
        targets.append(path)

    if not targets:
        print('No files to update.')
        return

    lang_embs = encode_texts(descriptions, model_name=args.model_name, device=args.device)

    for path, desc, emb in zip(targets, descriptions, lang_embs):
        data = torch.load(path)
        data.lang_emb = emb.detach().cpu()
        if args.add_text:
            data.lang_text = desc
        torch.save(data, path)

    print(f'Updated {len(targets)} samples with lang_emb.')


if __name__ == '__main__':
    main()
