"""CI regression gate for retrieval quality (fully offline, no LLM calls).

Runs the same offline ablation as retrieval_ablation.py and fails (non-zero
exit) if the production config's (+hybrid +reranking, RRF-boosted) Hit@1
drops below a floor set comfortably below the current measured baseline
(32.3%, ADR-004) but well above the pre-fix regressed configs (19.4% pure
reranker, 9.7% dense-only) — so it actually catches a real regression back
toward either of those, not just noise.
"""

import sys

from research_copilot.eval.retrieval_ablation import print_report, run_ablation

PRODUCTION_MODE = "+hybrid +reranking"
HIT_AT_1_FLOOR = 0.25


def main() -> int:
    results = run_ablation()
    print_report(results)

    hit_at_1 = results[PRODUCTION_MODE]["hit_rate@1"]
    print(f"\n{PRODUCTION_MODE} Hit@1: {hit_at_1:.1%} (floor: {HIT_AT_1_FLOOR:.0%})")

    if hit_at_1 < HIT_AT_1_FLOOR:
        print(f"REGRESSION: Hit@1 {hit_at_1:.1%} is below the {HIT_AT_1_FLOOR:.0%} floor.")
        return 1

    print("OK: no retrieval regression detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
