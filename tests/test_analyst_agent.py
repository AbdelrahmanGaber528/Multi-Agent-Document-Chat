from pathlib import Path
from typing import Any
import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.agents.analyst.agent import _AnalystSynthesis, run_analyst
from app.core.contract import EvidenceBundle, Finding

ROOT_DIR = Path(__file__).resolve().parents[1]
MOCK_EVIDENCE_PATH = ROOT_DIR / "data" / "mock_evidence.json"


class MockAnalystLLM:
    """Deterministic mock supporting `.bind_tools()` and `.with_structured_output()`."""

    def __init__(self, tool_turns: list[AIMessage], synthesis: _AnalystSynthesis) -> None:
        self._turns = list(tool_turns)
        self._synthesis = synthesis
        self.recorded_messages: list[list[BaseMessage]] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockAnalystLLM":
        self.bound_tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        self.recorded_messages.append(list(messages))
        if not self._turns:
            return AIMessage(content="")
        return self._turns.pop(0)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def _synthesize(messages: list[BaseMessage]) -> _AnalystSynthesis:
            self.recorded_messages.append(list(messages))
            return self._synthesis

        return RunnableLambda(_synthesize)


@pytest.fixture
def mock_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate_json(
        MOCK_EVIDENCE_PATH.read_text(encoding="utf-8")
    )


def test_empty_bundle_exits_early_without_llm_calls() -> None:
    empty_bundle = EvidenceBundle(query="Compare ResNet and ViT", chunks=[])
    llm = MockAnalystLLM(tool_turns=[], synthesis=_AnalystSynthesis())

    result = run_analyst(empty_bundle, llm)  # type: ignore[arg-type]

    assert llm.recorded_messages == []
    assert result.is_sufficient is False
    assert result.missing_info == ["Compare ResNet and ViT"]


def test_run_analyst_end_to_end_on_mock_evidence(mock_bundle: EvidenceBundle) -> None:
    turn_1 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "calculate",
                "args": {
                    "expression": "(2850 - 1200) / 1200 * 100",
                    "description": "ViT-B/16 vs ResNet-50 training cost increase (%)",
                    "chunk_ids": ["c2", "fake_c99"],
                },
            },
            {
                "id": "call_2",
                "name": "analyze_table_metric",
                "args": {
                    "chunk_id": "c1",
                    "metric_column": "Top-1 Accuracy",
                    "higher_is_better": True,
                },
            },
            {
                "id": "call_3",
                "name": "analyze_table_trend",
                "args": {
                    "chunk_id": "c4",
                    "period_column": "Year",
                    "metric_column": "Top-1 Accuracy",
                },
            },
            {
                "id": "call_4",
                "name": "analyze_table_correlation",
                "args": {
                    "chunk_id": "c1",
                    "x_column": "Params (M)",
                    "y_column": "Top-1 Accuracy",
                },
            },
            {
                "id": "call_5",
                "name": "compare_documents",
                "args": {
                    "topic": "ConvNeXt-T Across Papers",
                    "criteria": ["Top-1 Accuracy", "Latency (ms)", "Training Cost"],
                    "documents": [
                        {
                            "doc_id": "doc_1",
                            "attributes": {
                                "Top-1 Accuracy": "82.1%",
                                "Latency (ms)": "14.2",
                                "Training Cost": "$1,950",
                            },
                            "chunk_ids": ["c1", "c2"],
                        },
                        {
                            "doc_id": "doc_2",
                            "attributes": {
                                "Top-1 Accuracy": "82.1%",
                                "Latency (ms)": "14.2",
                                "Training Cost": "Not disclosed",
                            },
                            "chunk_ids": ["c3"],
                        },
                    ],
                },
            },
        ],
    )
    turn_2 = AIMessage(content="All tool analyses complete.")

    synthesis = _AnalystSynthesis(
        findings=[
            # Mixed real + fabricated chunk_ids -> "fake_c99" must be stripped
            Finding(
                statement="ConvNeXt-T achieves the highest Top-1 accuracy at 82.1%.",
                chunk_ids=["c1", "c3", "fake_c99"],
            ),
            # Only fabricated chunk_ids -> entire finding must be dropped
            Finding(
                statement="Unsupported claim from nonexistent chunk.",
                chunk_ids=["fake_c99"],
            ),
        ],
        missing_info=[],
    )

    llm = MockAnalystLLM(tool_turns=[turn_1, turn_2], synthesis=synthesis)
    result = run_analyst(mock_bundle, llm)  # type: ignore[arg-type]

    # Verify tool outputs were captured directly
    assert len(result.tables) == 2
    assert len(result.calculations) == 1
    assert result.calculations[0].result == 137.5
    assert result.calculations[0].chunk_ids == ["c2"]
    assert len(result.metric_analyses) == 3
    assert {a.analysis_type for a in result.metric_analyses} == {"ranking", "trend", "correlation"}
    assert len(result.comparisons) == 1

    # Verify citation grounding stripped hallucinated IDs everywhere
    assert len(result.findings) == 1
    assert result.findings[0].chunk_ids == ["c1", "c3"]
    assert "fake_c99" not in result.used_chunk_ids
    assert result.used_chunk_ids == ["c1", "c3", "c2", "c4"]

    # Verify feedback loop gap propagated from compare_documents
    assert result.is_sufficient is False
    assert result.missing_info == [
        "Training Cost of efficient_convnets_2026.pdf (ConvNeXt-T Across Papers)"
    ]


def test_tool_error_feedback_and_self_correction(mock_bundle: EvidenceBundle) -> None:
    # Turn 1: LLM uses wrong attribute key ("Accuracy" instead of "Top-1 Accuracy")
    bad_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "bad_call",
                "name": "compare_documents",
                "args": {
                    "topic": "ConvNeXt-T",
                    "criteria": ["Top-1 Accuracy"],
                    "documents": [
                        {"doc_id": "doc_1", "attributes": {"Accuracy": "82.1%"}, "chunk_ids": ["c1"]},
                        {"doc_id": "doc_2", "attributes": {"Accuracy": "82.1%"}, "chunk_ids": ["c3"]},
                    ],
                },
            }
        ],
    )
    # Turn 2: LLM reads the ToolMessage error and fixes the key
    fixed_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "good_call",
                "name": "compare_documents",
                "args": {
                    "topic": "ConvNeXt-T",
                    "criteria": ["Top-1 Accuracy"],
                    "documents": [
                        {"doc_id": "doc_1", "attributes": {"Top-1 Accuracy": "82.1%"}, "chunk_ids": ["c1"]},
                        {"doc_id": "doc_2", "attributes": {"Top-1 Accuracy": "82.1%"}, "chunk_ids": ["c3"]},
                    ],
                },
            }
        ],
    )
    done_turn = AIMessage(content="Done.")

    llm = MockAnalystLLM(
        tool_turns=[bad_turn, fixed_turn, done_turn],
        synthesis=_AnalystSynthesis(
            findings=[Finding(statement="Both papers report 82.1% accuracy.", chunk_ids=["c1", "c3"])],
            missing_info=[],
        ),
    )

    result = run_analyst(mock_bundle, llm)  # type: ignore[arg-type]

    # Confirm the error message was sent back to the LLM on turn 2
    second_turn_messages = llm.recorded_messages[1]
    tool_errors = [
        m for m in second_turn_messages if isinstance(m, ToolMessage) and m.content.startswith("Tool error:")
    ]
    assert len(tool_errors) == 1
    assert "must match criteria" in tool_errors[0].content

    # Confirm recovery succeeded and evidence is now sufficient
    assert len(result.comparisons) == 1
    assert result.is_sufficient is True


def test_attempted_queries_prevents_repeat_gaps(mock_bundle: EvidenceBundle) -> None:
    already_searched = "Training Cost of efficient_convnets_2026.pdf (ConvNeXt-T Across Papers)"

    turn_1 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_comp",
                "name": "compare_documents",
                "args": {
                    "topic": "ConvNeXt-T Across Papers",
                    "criteria": ["Top-1 Accuracy", "Training Cost"],
                    "documents": [
                        {
                            "doc_id": "doc_1",
                            "attributes": {"Top-1 Accuracy": "82.1%", "Training Cost": "$1,950"},
                            "chunk_ids": ["c1", "c2"],
                        },
                        {
                            "doc_id": "doc_2",
                            "attributes": {"Top-1 Accuracy": "82.1%", "Training Cost": "Not disclosed"},
                            "chunk_ids": ["c3"],
                        },
                    ],
                },
            }
        ],
    )
    turn_2 = AIMessage(content="Done.")

    llm = MockAnalystLLM(
        tool_turns=[turn_1, turn_2],
        synthesis=_AnalystSynthesis(
            findings=[Finding(statement="ConvNeXt-T achieves 82.1% accuracy.", chunk_ids=["c1", "c3"])],
            missing_info=[already_searched],
        ),
    )

    result = run_analyst(mock_bundle, llm, attempted_queries=[already_searched])  # type: ignore[arg-type]

    assert result.missing_info == []
    assert result.is_sufficient is True