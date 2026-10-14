"""scripts/deploy_hf_space.py -- Deploy SpeechLens Docker Space to Hugging Face.

Reads HF_TOKEN strictly from the environment; never writes or commits credentials.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def deploy(repo_id: str | None = None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Please set your Hugging Face write token: export HF_TOKEN='hf_...'", file=sys.stderr)
        print("Skipping deployment as per TASK_2_APP.md T12 fallback rules.", file=sys.stderr)
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub package not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return False

    api = HfApi(token=token)

    if not repo_id:
        user = api.whoami()["name"]
        repo_id = f"{user}/speechlens"

    print(f"Creating/updating Hugging Face Space: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    # Files and folders to upload
    upload_paths = [
        "Dockerfile",
        ".dockerignore",
        "requirements.lock",
        "pyproject.toml",
        "src",
        "api",
        "config",
        "data/library",
        "data/demo",
        "web/out",
    ]

    print("Uploading project assets to Space...")
    for rel_path in upload_paths:
        local_path = REPO_ROOT / rel_path
        if not local_path.exists():
            print(f"Warning: {rel_path} does not exist, skipping.")
            continue

        if local_path.is_file():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="space",
            )
        else:
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="space",
            )

    # Upload Space README.md
    space_readme = REPO_ROOT / "deploy" / "hf_space" / "README.md"
    if space_readme.exists():
        api.upload_file(
            path_or_fileobj=str(space_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"SUCCESS: Deployed to {space_url}")
    return True


if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else None
    success = deploy(target_repo)
    sys.exit(0 if success else 1)
