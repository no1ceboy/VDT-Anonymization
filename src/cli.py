"""Single command dispatcher for repository workflows."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "prepare-court": "src.data.prepare",
    "split-documents": "src.data.split_documents",
    "curate": "src.data.curate",
    "filter": "src.data.filtering",
    "filter-legacy-v1": "src.data.filter_legacy",
    "build-challenges": "src.data.challenges",
    "extract-vocabulary": "src.data.extract_vocabulary",
    "import-gazetteer": "src.data.import_gazetteer",
    "train-ocr-ngram": "src.pipeline.train_ocr_ngram",
    "run-ner": "src.pipeline.ner",
    "reconstruct": "src.pipeline.reconstruct",
    "kaggle-10k": "src.pipeline.kaggle",
    "review": "src.review.entity_links",
    "review-demo": "src.review.demo",
    "build-linking-data": "src.entity_linking.dataset",
    "build-raw-linking-data": "src.entity_linking.raw_dataset",
    "evaluate-linking-baseline": "src.entity_linking.baseline",
    "evaluate-linking-report-baseline": "src.entity_linking.report_baseline",
    "baseline-review": "src.review.baseline_cases",
    "train-linker": "src.entity_linking.training",
    "eval-linker": "src.entity_linking.evaluate",
    "eval-clusters": "src.entity_linking.cluster_evaluate",
    "link-documents": "src.entity_linking.inference",
    "anonymize": "src.local_cli",
    "serve-api": "src.api.app",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return (
        "Usage: python -m src.cli <command> [arguments]\n\n"
        f"Commands:\n{commands}\n\n"
        "Run the corresponding module under src with --help "
        "for command options."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return
    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(module_name)
    sys.argv = [f"python -m src.cli {command}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
