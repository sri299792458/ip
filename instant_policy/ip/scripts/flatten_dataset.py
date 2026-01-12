import argparse
import os
import shutil


def list_pt_files(input_root):
    files = []
    for root, _, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.startswith('data_') and fname.endswith('.pt'):
                files.append(os.path.join(root, fname))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root directory containing task subfolders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for flattened data_*.pt files')
    parser.add_argument('--mode', type=str, default='symlink', choices=['symlink', 'copy'],
                        help='Whether to symlink or copy files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing data_*.pt files in output_dir')
    parser.add_argument('--manifest', type=str, default='manifest.csv',
                        help='Manifest filename written in output_dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    existing = [f for f in os.listdir(args.output_dir) if f.startswith('data_') and f.endswith('.pt')]
    if existing and not args.overwrite:
        raise RuntimeError(f'Output dir already has {len(existing)} data_*.pt files. '
                           f'Remove them or pass --overwrite.')
    if existing and args.overwrite:
        for fname in existing:
            os.remove(os.path.join(args.output_dir, fname))

    files = list_pt_files(args.input_root)
    if not files:
        raise RuntimeError(f'No data_*.pt files found under {args.input_root}')

    manifest_path = os.path.join(args.output_dir, args.manifest)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('index,output_path,source_path\n')
        for idx, src in enumerate(files):
            dst = os.path.join(args.output_dir, f'data_{idx}.pt')
            if args.mode == 'symlink':
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            f.write(f'{idx},{dst},{src}\n')

    print(f'Flattened {len(files)} samples into {args.output_dir}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
