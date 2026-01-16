import argparse
import os
import shutil


def list_data_files(root_dirs):
    files = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("data_") and name.endswith(".pt"):
                    files.append(os.path.join(dirpath, name))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Merge pseudo-demo shards into a single directory.")
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = list_data_files(args.input_dirs)
    if not files:
        raise RuntimeError("No data_*.pt files found.")

    idx = args.start_index
    for src in files:
        dst = os.path.join(args.output_dir, f"data_{idx}.pt")
        if args.move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        idx += 1

    print(f"Merged {len(files)} files into {args.output_dir}")


if __name__ == "__main__":
    main()
