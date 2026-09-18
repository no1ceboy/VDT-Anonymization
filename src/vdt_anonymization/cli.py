"""Single command dispatcher for repository workflows."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "prepare-court": "vdt_anonymization.data.prepare",
    "curate": "vdt_anonymization.data.curate",
    "filter": "vdt_anonymization.data.filtering",
    "filter-legacy-v1": "vdt_anonymization.data.filter_legacy",
    "build-challenges": "vdt_anonymization.data.challenges",
    "extract-vocabulary": "vdt_anonymization.data.extract_vocabulary",
    "import-gazetteer": "vdt_anonymization.data.import_gazetteer",
    "run-ner": "vdt_anonymization.pipeline.ner",
    "reconstruct": "vdt_anonymization.pipeline.reconstruct",
    "kaggle-10k": "vdt_anonymization.pipeline.kaggle",
    "review": "vdt_anonymization.review.entity_links",
    "review-demo": "vdt_anonymization.review.demo",
    "build-linking-data": "vdt_anonymization.entity_linking.dataset",
    "build-raw-linking-data": "vdt_anonymization.entity_linking.raw_dataset",
    "evaluate-linking-baseline": "vdt_anonymization.entity_linking.baseline",
    "baseline-demo": "vdt_anonymization.entity_linking.baseline_demo",
    "train-linker": "vdt_anonymization.entity_linking.training",
    "eval-linker": "vdt_anonymization.entity_linking.evaluate",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return f"Usage: vdt <command> [arguments]\n\nCommands:\n{commands}\n\nRun 'vdt <command> --help' for command options."


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
    sys.argv = [f"vdt {command}", *sys.argv[2:]]
    module.main()
