"""scripts/deploy_hf_space.py -- Deploy SpeechLens Static Space to Hugging Face.

Uses space_sdk='static' (free tier) -- uploads the pre-built Next.js web/out export.
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

    print(f"Creating/updating Hugging Face Static Space: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
    )

    web_out = REPO_ROOT / "web" / "out"
    if not web_out.exists():
        print("ERROR: web/out does not exist.", file=sys.stderr)
        return False

    file_count = sum(1 for _ in web_out.rglob("*") if _.is_file())
    print(f"Uploading web/out static export ({file_count} files)...")
    api.upload_folder(
        folder_path=str(web_out),
        path_in_repo="",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["*.pyc", "__pycache__"],
        delete_patterns=["_next/static/chunks/app/page-*.js"],
    )

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
