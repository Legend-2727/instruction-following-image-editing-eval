import argparse
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload frozen 6K dataset artifacts to Hugging Face.")
    parser.add_argument("--repo-id", default="Legend2727/xLingual-picobanana-12k")
    parser.add_argument(
        "--metadata",
        default="data/hf_snapshots/xlingual_picobanana_multilingual_6k/metadata.jsonl",
    )
    parser.add_argument(
        "--labels",
        default="artifacts/multilingual_data_6k/labels.jsonl",
    )
    parser.add_argument("--path-prefix", default="frozen_6k")
    parser.add_argument("--create-pr", action="store_true")
    return parser.parse_args()


def upload_one(api: HfApi, repo_id: str, local_path: str, path_in_repo: str, create_pr: bool) -> None:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {path_in_repo}",
        create_pr=create_pr,
    )


def main() -> int:
    args = parse_args()
    api = HfApi()

    uploads = [
        (args.metadata, f"{args.path_prefix}/metadata.jsonl"),
        (args.labels, f"{args.path_prefix}/labels.jsonl"),
    ]

    print(f"Target dataset repo: {args.repo_id}")
    print(f"Mode: {'PR' if args.create_pr else 'direct push'}")

    for local_path, path_in_repo in uploads:
        try:
            print(f"Uploading {local_path} -> {path_in_repo}")
            upload_one(
                api=api,
                repo_id=args.repo_id,
                local_path=local_path,
                path_in_repo=path_in_repo,
                create_pr=args.create_pr,
            )
            print(f"Uploaded: {path_in_repo}")
        except Exception as exc:
            print(f"Failed uploading {path_in_repo}: {exc}")
            print("If direct push fails with 403, retry with --create-pr or use a write-scoped HF token.")
            return 1

    print("All uploads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
