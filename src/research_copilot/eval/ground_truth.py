"""Ground truth for the 41-question eval set (data/eval/questions_draft.md).

Each entry maps a question to the chunk(s) that actually contain the answer,
located by re-chunking the source .md files with the exact same parameters as
ingestion (research_copilot.chunking.chunk_document) and finding which chunk
contains a locator string pulled verbatim from the verified source quote.

Bucket 3 (adversarial/unanswerable) and Q25/Q30/Q40 (explicitly incomplete-
coverage or moved) are excluded — they don't have a single correct chunk to
locate, by design. They're evaluated separately in the citation-verification
pass (Task #5), not via retrieval precision@k.
"""

from pathlib import Path

from research_copilot.chunking import chunk_document
from research_copilot.config import settings

SEC_FILINGS_DIR = Path("data/sec_filings")

# (question_id, bucket, question_text, [(source_filename, locator_substring), ...])
# A question is "hit" if ANY of its listed chunk_ids appear in the retrieved set.
QUESTIONS = [
    (1, 1, "What were Western Alliance Bancorporation's total deposits at December 31, 2023?",
     [("WAL_10-K_2023-12-31.md", "Total deposits of $55.3 billion, up $1.7 billion")]),
    (2, 1, "What was Western Alliance's insured deposit ratio at December 31, 2023?",
     [("WAL_10-K_2023-12-31.md", "strengthened its insured deposit ratio from 45%")]),
    (3, 1, "What was Western Alliance's deposit exposure to the technology industry as of December 31, 2023?",
     [("WAL_10-K_2023-12-31.md", "deposit exposure to the technology industry totaled $4.4")]),
    (4, 1, "What were Comerica's total deposits at year-end 2023 vs. year-end 2022, on a period-end basis?",
     [("CMA_10-K_2023-12-31.md", "total deposits decreased $4.6 billion to $66.8 billion")]),
    (5, 1, "What was Comerica's diluted net income per common share in 2023 vs. 2022?",
     [("CMA_10-K_2023-12-31.md", "Diluted net income per common share was $6.44 in 2023")]),
    (6, 1, "How much did Comerica's net income decline in 2023, and to what level?",
     [("CMA_10-K_2023-12-31.md", "Net income decreased $270 million to $881 million")]),
    (7, 1, "What dividends did Comerica's subsidiary banks declare in 2023, 2022, and 2021?",
     [("CMA_10-K_2023-12-31.md", "declared dividends of $675 million in 2023")]),
    (8, 1, "What was Comerica's total uninsured deposits at year-end 2023 vs. year-end 2022?",
     [("CMA_10-K_2023-12-31.md", "Total uninsured deposits were $31.5 billion and $45.5 billion")]),
    (9, 1, "What was East West Bancorp's net income in 2023 vs. 2022?",
     [("EWBC_10-K_2023-12-31.md", "Net income | $ | 1,161,161 | $ | 1,128,083")]),
    (10, 1, "What were East West Bancorp's total assets, net loans, total deposits, and stockholders' equity at December 31, 2023?",
     [("EWBC_10-K_2023-12-31.md", "the Company had $69.6 billion in total assets")]),
    (11, 1, "What was East West Bancorp's CET1 capital ratio at December 31, 2023, vs. a year earlier?",
     [("EWBC_10-K_2023-12-31.md", "CET1 capital (to risk-weighted assets)")]),
    (12, 1, "What were Valley National Bancorp's total assets, net loans, deposits, and shareholders' equity at December 31, 2023?",
     [("VLY_10-K_2023-12-31.md", "Valley had consolidated total assets of $60.9 billion")]),
    (13, 1, "How many branches did Valley National have and how were they distributed by state at year-end 2023?",
     [("VLY_10-K_2023-12-31.md", "current 229 branch network")]),
    (14, 1, "What was M&T Bank's average total deposits in 2023, and what was the year-over-year change?",
     [("MTB_10-K_2023-12-31.md", "AVERAGE DEPOSITS")]),
    (15, 1, "What were M&T Bank's period-end core deposits at December 31 for 2023, 2022, and 2021?",
     [("MTB_10-K_2023-12-31.md", "Core deposits totaled $146.5 billion, $154.6 billion and $128.0 billion")]),
    (16, 1, "What was Zions Bancorporation's total deposit change in 2023 (full year), and what drove it?",
     [("ZION_10-K_2023-12-31.md", "Total deposits increased $3.3 billion, or 5%")]),
    (17, 2, "How did Zions' total deposits change differently across its operating segments in 2023?",
     [("ZION_10-K_2023-12-31.md", "Zions Bank’s income before income taxes decreased $76 million")]),
    (18, 2, "What was Western Alliance's uninsured deposit position at December 31, 2022, and how does it relate to the 45% insured ratio?",
     [("WAL_10-K_2022-12-31.md", "the Company had total uninsured deposits of $29.5 billion"),
      ("WAL_10-K_2023-12-31.md", "strengthened its insured deposit ratio from 45%")]),
    (19, 2, "How did Western Alliance's risk factors change between the FY2022 and FY2023 10-K regarding bank-failure/deposit-stress risk?",
     [("WAL_10-K_2023-12-31.md", "The bank closures in the first half of 2023 led to such disruption")]),
    (20, 2, "Trace Western Alliance's total deposits across every quarter of 2023 plus FY2022/FY2023 year-end figures.",
     [("WAL_10-K_2022-12-31.md", "Total deposits | 53,644 | 47,612"),
      ("WAL_10-Q_2023-03-31.md", "Total deposits | 47,587 | 53,644"),
      ("WAL_10-Q_2023-06-30.md", "Total deposits | 51,041 | 53,644"),
      ("WAL_10-Q_2023-09-30.md", "Total deposits | 54,287 | 53,644"),
      ("WAL_10-K_2023-12-31.md", "Total deposits of $55.3 billion, up $1.7 billion")]),
    (21, 2, "How did Zions' total deposits at March 31, 2023 compare to year-end 2022 and to March 2022?",
     [("ZION_10-Q_2023-03-31.md", "Total deposits decreased $13.1 billion, or 16%, to $69.2 billion")]),
    (22, 2, "Compare Comerica's and Western Alliance's period-end deposit trends in 2023.",
     [("WAL_10-K_2023-12-31.md", "Total deposits of $55.3 billion, up $1.7 billion"),
      ("CMA_10-K_2023-12-31.md", "total deposits decreased $4.6 billion to $66.8 billion")]),
    (23, 2, "What specific actions did Zions take during Q2 2023 in response to the regional banking stress?",
     [("ZION_10-Q_2023-06-30.md", "we managed the associated risks through the following actions")]),
    (24, 2, "Which bank had the larger year-over-year swing in shareholders' equity, Comerica or East West?",
     [("CMA_10-K_2023-12-31.md", "Total shareholders' equity increased $1.2 billion to $6.4 billion")]),
    (26, 2, "How does M&T Bank's declining Stress Capital Buffer relate to its CET1 capital requirement?",
     [("MTB_10-K_2023-12-31.md", "M&T's SCB of 4.0% became effective")]),
    (27, 2, "What acquisition explains most of the change in M&T's average core deposits between 2021 and 2022?",
     [("MTB_10-K_2023-12-31.md", "United acquisition added approximately $50.8 billion")]),
    (28, 2, "Compare WAL's insured-deposit-ratio recovery against Comerica's uninsured-deposit dollar trend.",
     [("WAL_10-K_2023-12-31.md", "strengthened its insured deposit ratio from 45%"),
      ("CMA_10-K_2023-12-31.md", "Total uninsured deposits were $31.5 billion and $45.5 billion")]),
    (29, 2, "How did Zions' loan-to-deposit ratio change from March 2022 to March 2023?",
     [("ZION_10-Q_2023-03-31.md", "Our loan-to-deposit ratio was 81%, compared with 62%")]),
    (31, 1, "What was East West Bancorp's exact deposit composition by category (checking, savings, "
            "time deposits, etc.) as a percentage breakdown at December 31, 2023?",
     [("EWBC_10-K_2023-12-31.md", "Noninterest-bearing demand | $ | 15,539,872 | 28")]),
    (38, 1, "What was M&T Bank's total deposits at December 31, 2023 (period-end, not average)?",
     [("MTB_10-K_2023-12-31.md", "Total deposits | 163,274 | 163,515")]),
    (41, 1, "What was Western Alliance's total deposits as of FY2023 10-K?",
     [("WAL_10-K_2023-12-31.md", "Total deposits of $55.3 billion, up $1.7 billion")]),
]

# Bucket 3 (adversarial/unanswerable) — no chunk_id to locate; expected behavior
# is abstention (no supported claims), evaluated by eval.citation_verification
# rather than retrieval precision@k. Numbering matches data/eval/questions_draft.md.
#
# Q31 and Q38 were REMOVED from this bucket (moved to QUESTIONS/bucket 1) after a
# direct read of the source filings found their claimed "unanswerable" premise was
# wrong: EWBC's deposit-mix percentage breakdown (Q31) IS present as a text table
# ("Noninterest-bearing demand ... 28%", etc. — not only as the embedded image
# originally cited), and MTB's period-end total deposits (Q38) IS a clean balance-
# sheet line item ("Total deposits | 163,274 | 163,515"), not the ambiguous/missing
# figure originally assumed. See KNOWN_TRADEOFFS.md and data/eval/questions_draft.md
# for the corrected write-up.
ADVERSARIAL_QUESTIONS = [
    (32, "What was Valley National Bancorp's workforce gender composition at year-end 2023?"),
    (33, "What was Silicon Valley Bank's total deposits at the time of its failure in March 2023?"),
    (34, "What were Western Alliance's total deposits at December 31, 2024?"),
    (35, "What was Zions Bancorporation's total deposits at December 31, 2021?"),
    (36, "What was Comerica's provision for credit losses in Q2 2023?"),
    (37, "Did Comerica report a net loss in any quarter of 2023?"),
    (39, "What was Western Alliance's total deposits at June 30, 2022?"),
    (40, "Which of the 6 banks in this corpus had the highest CET1 ratio at December 31, 2023?"),
]


def _find_chunk_index(
    source_file: str, locator: str, chunk_size_tokens: int, chunk_overlap_tokens: int
) -> int | None:
    path = SEC_FILINGS_DIR / source_file
    text = path.read_text()
    chunks = chunk_document(text, source_file, chunk_size_tokens, chunk_overlap_tokens)
    for chunk in chunks:
        if locator in chunk.text:
            return chunk.chunk_index
    return None


def build_ground_truth(
    chunk_size_tokens: int | None = None, chunk_overlap_tokens: int | None = None
) -> list[dict]:
    """Returns [{"id", "bucket", "question", "chunk_ids": [...]}], skipping any
    locator that couldn't be found (reported, not silently dropped).

    chunk_size/overlap default to the configured production values but can be
    overridden — used to test alternate chunking configs (e.g. a chunk-size
    ablation) without duplicating the question list."""
    chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
    chunk_overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens

    ground_truth = []
    for qid, bucket, question, sources in QUESTIONS:
        chunk_ids = []
        for source_file, locator in sources:
            idx = _find_chunk_index(
                source_file, locator, chunk_size_tokens, chunk_overlap_tokens
            )
            if idx is None:
                print(f"WARNING: Q{qid} locator not found in {source_file}: {locator!r}")
                continue
            chunk_ids.append(f"{source_file}#{idx}")
        if not chunk_ids:
            print(f"WARNING: Q{qid} has NO resolved ground-truth chunks — excluding from eval")
            continue
        ground_truth.append(
            {"id": qid, "bucket": bucket, "question": question, "chunk_ids": chunk_ids}
        )
    return ground_truth


if __name__ == "__main__":
    gt = build_ground_truth()
    print(f"\nResolved {len(gt)}/{len(QUESTIONS)} questions")
    for g in gt:
        print(f"Q{g['id']}: {g['chunk_ids']}")
