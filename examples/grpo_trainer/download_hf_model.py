import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download (test) a model from the Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="YongchengYAO/MedVision-V0-7B",
        help="Repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Directory to download into. Defaults to the shared HF cache.",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Optional glob patterns to fetch only matching files, e.g. "
        "--include 'config.json' '*.json' for a quick smoke test without the weights.",
    )
    args = parser.parse_args()

    # Auth is taken from the HF_TOKEN env var or the cached login token.
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=args.local_dir,
        allow_patterns=args.include,
    )
    print(f"Downloaded {args.repo_id} to:\n  {path}\n")

    total = 0
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            real = os.path.realpath(os.path.join(root, n))
            size = os.path.getsize(real)
            total += size
            files.append((os.path.relpath(os.path.join(root, n), path), size))

    for rel, size in sorted(files):
        print(f"  {size / 1e9:8.3f} GB  {rel}")
    print(f"\nTotal: {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
