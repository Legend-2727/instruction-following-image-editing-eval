from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Legend2727/xLingual-picobanana-12k"


class XlingualPicoBanana:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        local_dir: Optional[str] = None,
        download_images: bool = False,
    ):
        allow_patterns = ["metadata.jsonl"]
        if download_images:
            allow_patterns.append("images/*")

        if local_dir is not None:
            self.root = Path(local_dir)
        else:
            self.root = Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=allow_patterns,
                )
            )

        self.meta_path = self.root / "metadata.jsonl"
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict]:
        rows = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]

    def instruction(self, row: Dict, lang: str = "en") -> str:
        key = f"instruction_{lang}"
        if key not in row:
            raise KeyError(f"Missing language field: {key}")
        return row[key]

    def image_paths(self, row: Dict) -> Tuple[Path, Path]:
        return self.root / row["source_path"], self.root / row["target_path"]

    def available_languages(self) -> List[str]:
        if not self.rows:
            return []
        return sorted(
            k.replace("instruction_", "")
            for k in self.rows[0].keys()
            if k.startswith("instruction_")
        )
