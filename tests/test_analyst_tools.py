from pathlib import Path
import pytest

from app.agents.analyst.tools.calculator import CalculatorError, calculate
from app.agents.analyst.tools.data_analysis import (
    DataAnalysisError,
    analyze_numbers,
    analyze_table_correlation,
    analyze_table_metric,
    analyze_table_trend,
)
from app.agents.analyst.tools.document_comparison import (
    DocumentComparisonError,
    compare_documents,
    missing_to_info,
)
from app.agents.analyst.tools.table_extractor import (
    TableExtractorError,
    extract_from_chunks,
    extract_tables,
    get_column,
    parse_number,
)
from app.core.contract import DocumentEntry, EvidenceBundle

ROOT_DIR = Path(__file__).resolve().parents[1]
MOCK_EVIDENCE_PATH = ROOT_DIR / "data" / "mock_evidence.json"


@pytest.fixture
def mock_bundle() -> EvidenceBundle:
    raw = MOCK_EVIDENCE_PATH.read_text(encoding="utf-8")
    return EvidenceBundle.model_validate_json(raw)


# ---------------------------------------------------------------- 1. Mock Data
def test_mock_evidence_loads_cleanly(mock_bundle: EvidenceBundle) -> None:
    assert len(mock_bundle.chunks) == 4
    assert {c.doc_id for c in mock_bundle.chunks} == {"doc_1", "doc_2"}


# ---------------------------------------------------------------- 2. Calculator
def test_calculator_valid_and_unsafe() -> None:
    res = calculate("(2850 - 1200) / 1200 * 100", "ViT vs ResNet cost increase (%)", ["c2"])
    assert res.result == 137.5
    assert res.chunk_ids == ["c2"]

    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('ls')", "Malicious input", ["c1"])

    with pytest.raises(CalculatorError):
        calculate("10 / 0", "Zero division", ["c1"])


# ---------------------------------------------------------------- 3. Table Extractor
def test_table_extractor_on_mock_bundle(mock_bundle: EvidenceBundle) -> None:
    tables = extract_from_chunks(mock_bundle.chunks)
    assert len(tables) == 2
    assert tables[0].chunk_id == "c1"
    assert tables[1].chunk_id == "c4"

    acc_col = get_column(tables[0], "Top-1 Accuracy", numeric=True)
    assert acc_col == [76.1, 81.8, 82.1, 81.3]

    # Prose chunk returns [] without raising
    assert extract_tables(mock_bundle.chunks[1].text, "c2") == []


def test_parse_number_edge_cases() -> None:
    assert parse_number("92.5%") == 92.5
    assert parse_number("-$1,200") == -1200.0
    assert parse_number("(12.3)") == -12.3
    assert parse_number("0.85 ± 0.02") == 0.85
    assert parse_number("GPT-4") is None
    assert parse_number("N/A") is None


# ---------------------------------------------------------------- 4. Data Analysis
def test_data_analysis_ranking_trend_and_correlation(mock_bundle: EvidenceBundle) -> None:
    tables = extract_from_chunks(mock_bundle.chunks)
    bench_table, trend_table = tables[0], tables[1]

    # Higher is better
    acc_rank = analyze_table_metric(bench_table, "Top-1 Accuracy")
    assert acc_rank.details["best"]["labels"] == ["ConvNeXt-T"]
    assert acc_rank.details["best"]["value"] == 82.1

    # Lower is better
    lat_rank = analyze_table_metric(bench_table, "Latency (ms)", higher_is_better=False)
    assert lat_rank.details["best"]["labels"] == ["ResNet-50"]
    assert lat_rank.details["best"]["value"] == 12.4

    # Ties share rank 1
    tie_res = analyze_numbers([91.2, 91.2, 88.0], "F1", labels=["A", "B", "C"])
    assert tie_res.details["best"]["labels"] == ["A", "B"]
    assert [r["rank"] for r in tie_res.details["rankings"]] == [1, 1, 3]

    # Trend (c4: 2022 -> 2025)
    trend = analyze_table_trend(trend_table, "Year", "Top-1 Accuracy")
    assert trend.details["direction"] == "increasing"
    assert trend.details["change"] == 4.5
    assert trend.details["trend_is_favorable"] is True

    # Correlation (c1: Params vs Accuracy)
    corr = analyze_table_correlation(bench_table, "Params (M)", "Top-1 Accuracy")
    assert corr.analysis_type == "correlation"
    assert "Very few points" in corr.summary


# ---------------------------------------------------------------- 5. Document Comparison
def test_document_comparison_and_feedback_gap(mock_bundle: EvidenceBundle) -> None:
    comp = compare_documents(
        topic="ConvNeXt-T Results Across Papers",
        criteria=["Top-1 Accuracy", "Latency (ms)", "Training Cost"],
        documents=[
            DocumentEntry(
                doc_id="doc_1",
                attributes={
                    "Top-1 Accuracy": "82.1%",
                    "Latency (ms)": "14.2",
                    "Training Cost": "$1,950",
                },
                chunk_ids=["c1", "c2"],
            ),
            DocumentEntry(
                doc_id="doc_2",
                attributes={
                    "Top-1 Accuracy": "82.1%",
                    "Latency (ms)": "14.2",
                    "Training Cost": "Not disclosed",
                },
                chunk_ids=["c3"],
            ),
        ],
        chunks=mock_bundle.chunks,
    )

    assert comp.missing_by_doc == {"doc_2": ["Training Cost"]}
    gaps = missing_to_info(comp)
    assert gaps == ["Training Cost of efficient_convnets_2026.pdf (ConvNeXt-T Results Across Papers)"]

    # Rejects invented chunk_id
    with pytest.raises(DocumentComparisonError, match="Unknown chunk_id"):
        compare_documents(
            topic="Test",
            criteria=["Accuracy"],
            documents=[
                DocumentEntry(doc_id="doc_1", attributes={"Accuracy": "82%"}, chunk_ids=["fake_id"]),
                DocumentEntry(doc_id="doc_2", attributes={"Accuracy": "82%"}, chunk_ids=["c3"]),
            ],
            chunks=mock_bundle.chunks,
        )