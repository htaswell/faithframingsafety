"""Command-line entry point.

    python -m adviceaudit <subcommand> [options]

Subcommands
-----------
count     Pipeline A step 1: lexicon keyword counts per response.
fisher    Pipeline A step 2: Fisher's exact tests on keyword presence.
annotate  Pipeline B step 1: ordinal 0-3 annotation by an LLM judge.
ordinal   Pipeline B step 2: Mann-Whitney U tests on annotation scores.
figures   Optional: score-distribution figures for the ordinal pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_ANALYSIS_CONFIG = "config/analysis.yaml"
DEFAULT_LEXICON_CONFIG = "config/lexicon.yaml"
DEFAULT_RUBRIC_CONFIG = "config/rubric.yaml"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input table (.csv/.tsv/.xlsx)")
    parser.add_argument("--output", required=True, help="Output table (.csv/.tsv/.xlsx)")
    parser.add_argument(
        "--analysis-config",
        default=DEFAULT_ANALYSIS_CONFIG,
        help=f"Path to analysis config (default: {DEFAULT_ANALYSIS_CONFIG})",
    )
    parser.add_argument(
        "--sheet", default=0, help="Sheet name or index for Excel inputs (default: 0)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adviceaudit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Pipeline A -------------------------------------------------------
    count = subparsers.add_parser("count", help="Lexicon keyword counts per response")
    _add_common(count)
    count.add_argument("--lexicon-config", default=DEFAULT_LEXICON_CONFIG)

    fisher = subparsers.add_parser("fisher", help="Fisher's exact tests on keyword presence")
    _add_common(fisher)
    fisher.add_argument("--lexicon-config", default=DEFAULT_LEXICON_CONFIG)
    fisher.add_argument("--excel-output", default=None, help="Optional .xlsx workbook")

    # ---- Pipeline B -------------------------------------------------------
    annotate = subparsers.add_parser("annotate", help="Ordinal 0-3 LLM annotation")
    _add_common(annotate)
    annotate.add_argument("--rubric-config", default=DEFAULT_RUBRIC_CONFIG)
    annotate.add_argument("--cache", default=None, help="Path to the JSONL annotation cache")
    annotate.add_argument(
        "--mock",
        action="store_true",
        help="Generate deterministic placeholder scores with no API access. "
        "For testing the pipeline only; never use for reported results.",
    )
    annotate.add_argument(
        "--limit", type=int, default=None, help="Annotate only the first N rows"
    )
    annotate.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel API calls (overrides annotation.max_workers in the config)",
    )

    ordinal = subparsers.add_parser("ordinal", help="Mann-Whitney U tests on annotation scores")
    _add_common(ordinal)
    ordinal.add_argument("--rubric-config", default=DEFAULT_RUBRIC_CONFIG)
    ordinal.add_argument("--distribution-output", default=None, help="Optional score-distribution table")
    ordinal.add_argument("--excel-output", default=None, help="Optional .xlsx workbook")

    # ---- Figures ----------------------------------------------------------
    figures = subparsers.add_parser("figures", help="Score-distribution figures")
    figures.add_argument("--input", required=True, help="Annotated table")
    figures.add_argument("--output-dir", required=True, help="Directory for figures")
    figures.add_argument("--analysis-config", default=DEFAULT_ANALYSIS_CONFIG)
    figures.add_argument("--rubric-config", default=DEFAULT_RUBRIC_CONFIG)
    figures.add_argument("--sheet", default=0)
    figures.add_argument("--format", default="png", choices=["png", "pdf", "svg"])

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    if args.command == "count":
        from . import lexicon_counts

        result = lexicon_counts.run(
            input_path=args.input,
            output_path=args.output,
            analysis_config_path=args.analysis_config,
            lexicon_config_path=args.lexicon_config,
            sheet=sheet,
        )
        print(f"Counted {len(result)} rows -> {args.output}")

    elif args.command == "fisher":
        from . import fishers

        result = fishers.run(
            input_path=args.input,
            output_path=args.output,
            analysis_config_path=args.analysis_config,
            lexicon_config_path=args.lexicon_config,
            sheet=sheet,
            excel_output=args.excel_output,
        )
        n_sig = int(result["significant_adj"].fillna(False).sum())
        print(f"{len(result)} comparisons, {n_sig} significant after BH correction "
              f"-> {args.output}")

    elif args.command == "annotate":
        from . import annotate as annotate_mod

        result = annotate_mod.run(
            input_path=args.input,
            output_path=args.output,
            analysis_config_path=args.analysis_config,
            rubric_config_path=args.rubric_config,
            sheet=sheet,
            cache_path=args.cache,
            mock=args.mock,
            limit=args.limit,
            max_workers=args.max_workers,
        )
        print(f"Annotated {len(result)} rows -> {args.output}")

    elif args.command == "ordinal":
        from . import mann_whitney

        result = mann_whitney.run(
            input_path=args.input,
            output_path=args.output,
            analysis_config_path=args.analysis_config,
            rubric_config_path=args.rubric_config,
            sheet=sheet,
            distribution_output=args.distribution_output,
            excel_output=args.excel_output,
        )
        n_meaningful = int(result["meaningful_adj"].fillna(False).sum())
        print(f"{len(result)} comparisons, {n_meaningful} meaningful after BH correction "
              f"-> {args.output}")

    elif args.command == "figures":
        from .io_utils import load_config, read_table
        from .plots import plot_score_distributions

        analysis_cfg = load_config(args.analysis_config)
        rubric = load_config(args.rubric_config)
        columns = analysis_cfg["columns"]
        identities = analysis_cfg["identities"]
        score_levels = sorted(int(level) for level in rubric.get("scale", {}))
        paths = plot_score_distributions(
            read_table(args.input, sheet=sheet),
            columns=columns,
            metrics=list(rubric["dimensions"]),
            identities=identities,
            score_levels=score_levels,
            output_dir=args.output_dir,
            file_format=args.format,
        )
        print(f"Wrote {len(paths)} figures -> {Path(args.output_dir)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
