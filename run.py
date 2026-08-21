"""Project-level entry point for the current CUMCM 2025 Problem A checkpoint."""

import json
from pathlib import Path

from src.q5 import run_question_5

import mmkit
from mmkit.artifacts import create_run_manifest, prepare_output_paths, write_manifest
from mmkit.experiments import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parent
PROBLEM_PDF = PROJECT_ROOT / "data/raw/CUMCM-2025-problem+A-Chinese.pdf"
Q5_CONFIG = PROJECT_ROOT / "configs/q5.json"


def main() -> None:
    """Run the currently approved Question 5 checkpoint and write its manifest."""
    output_paths = prepare_output_paths(PROJECT_ROOT / "outputs")
    config = json.loads(Q5_CONFIG.read_text(encoding="utf-8"))
    seed_state = set_random_seed(int(config["seed"]))
    artifacts = run_question_5(PROJECT_ROOT, config)
    manifest = create_run_manifest(
        run_name=PROJECT_ROOT.name,
        seed=seed_state.seed,
        config={"status": "question-5", "problem": "CUMCM-2025-A", "q5": config},
        input_files=[PROBLEM_PDF, Q5_CONFIG],
        artifacts=artifacts,
    )
    for input_file in manifest["input_files"]:
        input_file["path"] = Path(input_file["path"]).relative_to(PROJECT_ROOT).as_posix()
    manifest["artifacts"] = [
        Path(artifact).relative_to(PROJECT_ROOT).as_posix()
        for artifact in manifest["artifacts"]
    ]
    manifest_path = write_manifest(
        manifest,
        output_paths.root / "manifest.json",
        overwrite=True,
    )
    print(f"Question 5 complete; mmkit={mmkit.__version__}; manifest={manifest_path}")


if __name__ == "__main__":
    main()
