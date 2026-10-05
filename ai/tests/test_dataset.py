"""Dataset Evaluation Benchmark.

Evaluates the NLP pipeline against data/dataset.csv measuring:
1. Sentence classification accuracy
2. ACTION precision / recall / F1
3. DECISION precision / recall / F1
4. DISCUSSION precision / recall / F1
5. INFORMATION precision / recall / F1
6. UNKNOWN precision / recall / F1
7. Responsible-person extraction accuracy
8. Deadline extraction accuracy
9. Deadline normalization accuracy
10. Decision extraction accuracy
11. Status extraction accuracy
"""

import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from models.schemas import SentenceType, TaskStatus
from nlp.adapters import from_csv_row
from nlp.pipeline import process_segment


def compute_metrics(dataset_path: str = "data/dataset.csv"):
    # Reference date for deterministic evaluation: 2026-10-05 (Monday)
    ref_date = datetime(2026, 10, 5, 10, 0, 0)

    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(p, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    if total == 0:
        raise ValueError("Dataset is empty")

    correct_classifications = 0

    classes = [
        SentenceType.ACTION.value,
        SentenceType.DECISION.value,
        SentenceType.DISCUSSION.value,
        SentenceType.INFORMATION.value,
        SentenceType.UNKNOWN.value,
    ]

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    person_total = 0
    person_correct = 0

    deadline_text_total = 0
    deadline_text_correct = 0

    deadline_norm_total = 0
    deadline_norm_correct = 0

    decision_total = 0
    decision_correct = 0

    status_total = 0
    status_correct = 0

    for row in reader:
        gold_type = row["type"].strip().upper()
        gold_person = row["responsible_person"].strip() or None
        gold_deadline_text = row["deadline_text"].strip() or None
        gold_deadline_norm = row["deadline_normalized"].strip() or None
        gold_decision = row["decision"].strip() or None
        gold_status = row["status"].strip() or "Unknown"

        inp = from_csv_row(row)
        pred = process_segment(inp, reference_datetime=ref_date)
        pred_type = pred.type.value if hasattr(pred.type, "value") else str(pred.type)
        pred_status = pred.status.value if hasattr(pred.status, "value") else str(pred.status)

        # Classification metrics
        if pred_type == gold_type:
            correct_classifications += 1
            tp[gold_type] += 1
        else:
            fp[pred_type] += 1
            fn[gold_type] += 1

        # Action-specific field evaluations
        if gold_type == SentenceType.ACTION.value:
            # Responsible Person
            if gold_person is not None:
                person_total += 1
                if pred.responsible_person and pred.responsible_person.strip().lower() == gold_person.lower():
                    person_correct += 1

            # Deadline text
            if gold_deadline_text is not None:
                deadline_text_total += 1
                if pred.deadline_text and pred.deadline_text.strip().lower() == gold_deadline_text.lower():
                    deadline_text_correct += 1

            # Deadline normalized
            if gold_deadline_norm is not None:
                deadline_norm_total += 1
                if pred.deadline_normalized == gold_deadline_norm:
                    deadline_norm_correct += 1

            # Status
            status_total += 1
            if pred_status.lower() == gold_status.lower():
                status_correct += 1

        # Decision-specific evaluation
        if gold_type == SentenceType.DECISION.value:
            if gold_decision is not None:
                decision_total += 1
                if pred.decision and pred.decision.strip().lower() == gold_decision.lower():
                    decision_correct += 1

    # Overall accuracy
    accuracy = correct_classifications / total

    # Per-class P / R / F1
    prf1 = {}
    for c in classes:
        true_pos = tp[c]
        false_pos = fp[c]
        false_neg = fn[c]
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        prf1[c] = {"precision": precision, "recall": recall, "f1": f1}

    person_acc = person_correct / person_total if person_total > 0 else 1.0
    deadline_text_acc = deadline_text_correct / deadline_text_total if deadline_text_total > 0 else 1.0
    deadline_norm_acc = deadline_norm_correct / deadline_norm_total if deadline_norm_total > 0 else 1.0
    decision_acc = decision_correct / decision_total if decision_total > 0 else 1.0
    status_acc = status_correct / status_total if status_total > 0 else 1.0

    return {
        "total_samples": total,
        "classification_accuracy": accuracy,
        "class_metrics": prf1,
        "person_extraction_accuracy": person_acc,
        "deadline_text_accuracy": deadline_text_acc,
        "deadline_norm_accuracy": deadline_norm_acc,
        "decision_extraction_accuracy": decision_acc,
        "status_extraction_accuracy": status_acc,
    }


def print_report(results: dict):
    print("\n" + "=" * 65)
    print("        MEETING DECISION EXTRACTOR — DATASET EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Evaluated Samples: {results['total_samples']}")
    print(f"1. Sentence Classification Accuracy: {results['classification_accuracy']:.2%}\n")

    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    for c, m in results["class_metrics"].items():
        print(f"{c:<14} | {m['precision']:<10.2%} | {m['recall']:<10.2%} | {m['f1']:<10.2%}")

    print("\n" + "-" * 55)
    print("Field Extraction Accuracies:")
    print(f"7.  Responsible Person Extraction Accuracy: {results['person_extraction_accuracy']:.2%}")
    print(f"8.  Deadline Extraction (Text) Accuracy:    {results['deadline_text_accuracy']:.2%}")
    print(f"9.  Deadline Normalization Accuracy:        {results['deadline_norm_accuracy']:.2%}")
    print(f"10. Decision Extraction Accuracy:           {results['decision_extraction_accuracy']:.2%}")
    print(f"11. Status Extraction Accuracy:             {results['status_extraction_accuracy']:.2%}")
    print("=" * 65 + "\n")


def test_dataset_evaluation_thresholds():
    """Pytest test to ensure metrics meet required quality thresholds."""
    results = compute_metrics("data/dataset.csv")
    print_report(results)
    assert results["classification_accuracy"] >= 0.90, "Classification accuracy below 90%"
    assert results["person_extraction_accuracy"] >= 0.85, "Person extraction accuracy below 85%"
    assert results["decision_extraction_accuracy"] >= 0.85, "Decision extraction accuracy below 85%"


if __name__ == "__main__":
    res = compute_metrics()
    print_report(res)
