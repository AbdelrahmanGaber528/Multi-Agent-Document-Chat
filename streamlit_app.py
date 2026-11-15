import os
import json
import logging
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.core.config import get_settings
from app.core.contract import (
    AnalysisResult,
    EvidenceBundle,
    EvidenceChunk,
    FinalAnswer,
    Finding,
    Calculation,
    ExtractedTable,
    MetricAnalysis,
    Comparison,
    Citation,
)
from app.core.errors import AppError
from app.pipeline.ingestion import ingest_file
from app.pipeline.vector_db import VectorDatabase
from app.agents.retriever.agent import RetrieverAgent
from app.agents.analyst.agent import _AnalystSynthesis
from app.orchestrator.graph import build_graph
from app.orchestrator.state import create_initial_state

logger = logging.getLogger(__name__)

# ── App Configuration & Styling ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Document Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Metric & Card styling */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    .agent-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-retriever { background-color: #2b4c7e; color: #d0e1fd; }
    .badge-analyst { background-color: #5c3c86; color: #ebd8fd; }
    .badge-answer { background-color: #1e6347; color: #c6f6d5; }
    .badge-feedback { background-color: #7b4a12; color: #feebc8; }
    
    .citation-card {
        border-left: 3px solid #3182ce;
        background-color: rgba(49, 130, 206, 0.08);
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .finding-card {
        border-left: 3px solid #805ad5;
        background-color: rgba(128, 90, 213, 0.08);
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .calc-card {
        border-left: 3px solid #38a169;
        background-color: rgba(56, 161, 105, 0.08);
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing vector database and embedding model...")
def get_db():
    settings = get_settings()
    settings.ensure_dirs()
    return VectorDatabase(embedding_model=settings.embedding_model)


def get_indexed_documents(db: VectorDatabase) -> list[dict[str, Any]]:
    try:
        stored = db.vector_store.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for m in stored.get("metadatas") or []:
            if not m:
                continue
            doc_id = m.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": m.get("doc_name") or m.get("filename") or "Unknown Document",
                    "chunks": 0,
                }
            docs[doc_id]["chunks"] += 1
        return list(docs.values())
    except Exception as exc:
        logger.warning("Could not list documents: %s", exc)
        return []


def load_demo_mock_data(db: VectorDatabase) -> int:
    """Loads sample vision benchmark evidence into Chroma for instant testing."""
    root_dir = Path(__file__).resolve().parent
    mock_file = root_dir / "data" / "mock_evidence.json"
    if not mock_file.exists():
        return 0

    with open(mock_file, encoding="utf-8") as f:
        data = json.load(f)

    from langchain_core.documents import Document

    docs = []
    for chunk in data.get("chunks", []):
        docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "doc_name": chunk["doc_name"],
                    "filename": chunk["doc_name"],
                    "source": chunk["doc_name"],
                    "page": chunk.get("page"),
                    "score": chunk.get("score", 0.9),
                    "chunk_type": chunk.get("chunk_type", "text"),
                },
            )
        )
    if docs:
        ids = db.add_documents(docs)
        return len(ids)
    return 0


# ── Mock LLM for Demo Mode ────────────────────────────────────────────────────
class DemoAnalystLLM:
    """Deterministic Mock LLM that runs sample analysis on vision benchmarks without an API key."""

    def __init__(self):
        self._step = 0

    def bind_tools(self, tools: list[Any], **kwargs: Any):
        return self

    def invoke(self, messages: list[Any], **kwargs: Any):
        # Round 1: Call tools or finish
        return AIMessage(content="", tool_calls=[])

    def with_structured_output(self, schema: Any, **kwargs: Any):
        def _synth(messages: list[Any]) -> _AnalystSynthesis:
            return _AnalystSynthesis(
                findings=[
                    Finding(
                        statement="ConvNeXt-T achieves 82.1% Top-1 accuracy with 14.2 ms latency and $1,950 training cost (vision_benchmarks_2025.pdf).",
                        chunk_ids=["c1", "c2"],
                    ),
                    Finding(
                        statement="ResNet-50 achieves 76.1% accuracy with 12.4 ms latency and $1,200 training cost.",
                        chunk_ids=["c1", "c2"],
                    ),
                    Finding(
                        statement="Annual best ConvNet accuracy progresses steadily from 78.4% in 2022 to 82.9% in 2025 (efficient_convnets_2026.pdf).",
                        chunk_ids=["c4"],
                    ),
                ],
                missing_info=[],
            )

        return RunnableLambda(_synth)


class DemoAnswerLLM:
    def invoke(self, messages: list[Any], **kwargs: Any):
        content = (
            "### Architecture Comparison\n\n"
            "| Model | Params (M) | Top-1 Accuracy | Latency (ms) | Training Cost |\n"
            "| :--- | :---: | :---: | :---: | :---: |\n"
            "| **ConvNeXt-T** | 28.6 | 82.1% [c1] | 14.2 ms [c1] | $1,950 [c2] |\n"
            "| **ViT-B/16** | 86.6 | 81.8% [c1] | 29.5 ms [c1] | $2,850 [c2] |\n"
            "| **ResNet-50** | 25.6 | 76.1% [c1] | 12.4 ms [c1] | $1,200 [c2] |\n"
            "| **Swin-T** | 28.3 | 81.3% [c1] | 18.7 ms [c1] | — |\n\n"
            "### Key Quantitative Insights\n\n"
            "1. **Accuracy vs. Latency Tradeoff**: ConvNeXt-T outperforms ResNet-50 by +6.0 percentage points in Top-1 Accuracy with only a 1.8 ms increase in latency [c1].\n"
            "2. **Cost Efficiency**: Training ViT-B/16 costs $2,850 (+46.2% more expensive than ConvNeXt-T) despite achieving lower accuracy [c1][c2].\n"
            "3. **Annual Progress**: From 2022 to 2025, annual ConvNet Top-1 accuracy improved by +4.5 percentage points (78.4% → 82.9%) [c4]."
        )
        return AIMessage(content=content)


# ── Execution Pipeline ────────────────────────────────────────────────────────
def execute_query(
    question: str,
    history: list[dict[str, Any]],
    max_rounds: int,
    use_demo_mode: bool,
    api_key: Optional[str],
    model_name: str,
    base_url: Optional[str],
    temperature: float,
) -> tuple[FinalAnswer, dict[str, Any]]:
    db = get_db()
    settings = get_settings()

    if use_demo_mode:
        analyst_llm = DemoAnalystLLM()
        answer_llm = DemoAnswerLLM()
        retriever = RetrieverAgent(llm=DemoAnswerLLM(), vector_db=db)
    else:
        if not api_key:
            raise ValueError("LLM API Key is required when not in Demo Mode.")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url or None,
            temperature=temperature,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        retriever = RetrieverAgent(llm=llm, vector_db=db)
        analyst_llm = llm
        answer_llm = llm

    workflow = build_graph(
        retrieve_evidence=lambda q, hist: retriever.invoke(q),
        analyst_llm=analyst_llm,
        answer_llm=answer_llm,
    )

    initial_state = create_initial_state(
        question=question,
        history=history or [],
        max_rounds=max_rounds,
    )

    final_state = workflow.invoke(
        initial_state,
        config={"recursion_limit": max(25, 2 * max_rounds + 5)},
    )

    final_answer: FinalAnswer = final_state["final_answer"]
    return final_answer, final_state


# ── UI Layout ─────────────────────────────────────────────────────────────────
settings = get_settings()
db = get_db()

# Sidebar: Document Management & Settings
with st.sidebar:
    st.title("⚙️ System Control")

    # Document Management Section
    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Process & Ingest Files", use_container_width=True):
            success_count = 0
            with st.spinner("Ingesting and chunking documents..."):
                for uf in uploaded_files:
                    try:
                        res = ingest_file(uf.name, uf.getvalue())
                        st.success(f"✓ {uf.name} ({res.num_chunks} chunks)")
                        success_count += 1
                    except Exception as exc:
                        st.error(f"Error ingesting {uf.name}: {exc}")
            if success_count > 0:
                st.rerun()

    # Document Inventory
    current_docs = get_indexed_documents(db)
    total_chunks = sum(d["chunks"] for d in current_docs)

    col1, col2 = st.columns(2)
    col1.metric("Documents", len(current_docs))
    col2.metric("Total Chunks", total_chunks)

    # Demo Benchmark Button
    if len(current_docs) == 0:
        st.info("💡 No documents yet. Load sample vision benchmarks to test immediately!")
        if st.button("📥 Load Sample Benchmark Data", use_container_width=True):
            with st.spinner("Loading benchmark evidence..."):
                count = load_demo_mock_data(db)
                st.success(f"Loaded {count} benchmark chunks!")
                st.rerun()

    if current_docs:
        with st.expander("📄 Document Inventory", expanded=False):
            for d in current_docs:
                dcol1, dcol2 = st.columns([3, 1])
                dcol1.markdown(f"**{d['filename']}**  \n`{d['chunks']} chunks`")
                if dcol2.button("🗑️", key=f"del_{d['doc_id']}", help="Delete document"):
                    db.delete_document(d["doc_id"])
                    st.rerun()

            if st.button("Clear All Documents", type="secondary", use_container_width=True):
                for d in current_docs:
                    db.delete_document(d["doc_id"])
                st.rerun()

    st.divider()

    # Agent / LLM Settings
    st.subheader("🧠 Multi-Agent Configuration")
    demo_mode = st.toggle(
        "Offline / Demo Mode",
        value=not bool(settings.llm_api_key),
        help="Use deterministic mock agents for testing without an active API key.",
    )

    if demo_mode:
        st.caption("ℹ️ Demo Mode active: queries will run against benchmark heuristics.")
        api_key = None
        model_name = "demo-agent"
        base_url = None
        temperature = 0.0
    else:
        api_key = st.text_input(
            "LLM API Key",
            value=settings.llm_api_key or "",
            type="password",
            help="OpenRouter / OpenAI / Anthropic-compatible API key.",
        )
        model_name = st.text_input(
            "Model Name",
            value=settings.llm_model or "openrouter/free",
            help="e.g. openrouter/free, google/gemini-2.0-flash-exp:free, openai/gpt-4o-mini",
        )
        base_url = st.text_input(
            "Base URL",
            value=settings.llm_base_url or "https://openrouter.ai/api/v1",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, float(settings.llm_temperature), 0.05)

    max_rounds = st.slider(
        "Max Feedback Rounds",
        min_value=1,
        max_value=5,
        value=int(settings.max_rounds),
        help="Maximum loops between Analyst and Retriever to fill evidence gaps.",
    )

    st.divider()
    with st.expander("ℹ️ Multi-Agent Architecture", expanded=False):
        st.markdown(
            """
            - **🔍 Retriever Agent**: Hybrid semantic (Chroma) + keyword BM25 search with Reciprocal Rank Fusion & token budget pruning.
            - **🧠 Analyst Agent**: Fact verification using safe arithmetic, table parser, statistics, trends, and cross-doc matrix.
            - **🔄 Feedback Loop**: Detects missing facts and dynamically re-queries the Retriever.
            - **✍️ Answer Agent**: Generates grounded synthesis with inline citation verification.
            """
        )

# Main Header
st.title("🤖 Multi-Agent Document Chat")
st.markdown(
    "Ask complex multi-document questions with automated **table extraction**, **quantitative calculation**, and **strict source verification**."
)

# Tabs
tab_chat, tab_explorer, tab_tools = st.tabs(["💬 Chat & Trace", "📚 Evidence Explorer", "🧪 Tool Sandbox"])

# ── Tab 1: Chat & Multi-Agent Trace ───────────────────────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick prompt pills
    st.markdown("**Sample Inquiries:**")
    prompt_cols = st.columns(3)
    p1 = prompt_cols[0].button("📊 ConvNeXt-T vs ResNet-50", use_container_width=True)
    p2 = prompt_cols[1].button("📈 Accuracy Trend 2022-2025", use_container_width=True)
    p3 = prompt_cols[2].button("💰 Training Cost & Parameters", use_container_width=True)

    selected_prompt = None
    if p1:
        selected_prompt = "Compare the Top-1 accuracy, latency, and training cost of ConvNeXt-T and ResNet-50."
    elif p2:
        selected_prompt = "What is the annual progression of ConvNet Top-1 accuracy from 2022 to 2025, and what was the total improvement?"
    elif p3:
        selected_prompt = "How does training cost and parameter count compare between ViT-B/16 and ConvNeXt-T?"

    # Display Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains multi-agent trace details, show them in an expander
            if "trace" in msg and msg["trace"]:
                trace = msg["trace"]
                with st.expander("🔍 View Multi-Agent Execution Trace", expanded=False):
                    rounds = trace.get("rounds_used", 1)
                    st.markdown(
                        f"<span class='agent-badge badge-answer'>Rounds Used: {rounds}</span>"
                        f"<span class='agent-badge badge-retriever'>Chunks Retrieved: {len(trace.get('chunks_by_id', {}))}</span>",
                        unsafe_allow_html=True,
                    )

                    analysis = trace.get("analysis")
                    if analysis:
                        # Findings
                        if analysis.findings:
                            st.markdown("##### 📌 Verified Findings")
                            for f in analysis.findings:
                                cids = ", ".join(f.chunk_ids)
                                st.markdown(
                                    f"<div class='finding-card'><strong>{f.statement}</strong><br><small style='color:gray;'>Chunks: {cids}</small></div>",
                                    unsafe_allow_html=True,
                                )

                        # Calculations
                        if analysis.calculations:
                            st.markdown("##### 🧮 Tool-Computed Arithmetic")
                            for c in analysis.calculations:
                                st.markdown(
                                    f"<div class='calc-card'><strong>{c.description}</strong>: <code>{c.expression}</code> = <b>{c.result}</b></div>",
                                    unsafe_allow_html=True,
                                )

                        # Metric Analyses (Rankings, Trends, Correlations)
                        if getattr(analysis, "metric_analyses", None):
                            st.markdown("##### 📊 Metric Analyses")
                            for m in analysis.metric_analyses:
                                st.info(f"**[{m.analysis_type.upper()}] {m.metric}**: {m.summary}")

                        # Document Comparisons
                        if analysis.comparisons:
                            st.markdown("##### 📑 Cross-Document Comparison")
                            for comp in analysis.comparisons:
                                st.markdown(f"**Topic**: {comp.topic}")
                                if comp.similarities:
                                    st.markdown("**Similarities**:")
                                    for s in comp.similarities:
                                        st.markdown(f"- {s}")
                                if comp.differences:
                                    st.markdown("**Differences**:")
                                    for d in comp.differences:
                                        st.markdown(f"- {d}")

                    # Sources / Citations
                    citations = trace.get("citations", [])
                    if citations:
                        st.markdown("##### 📚 Source Citations")
                        for cit in citations:
                            page_str = f", Page {cit.page}" if cit.page else ""
                            st.markdown(
                                f"<div class='citation-card'><b>{cit.doc_name}</b>{page_str} (Chunk <code>{cit.chunk_id}</code>)</div>",
                                unsafe_allow_html=True,
                            )

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the uploaded documents...") or selected_prompt

    if user_query:
        if not demo_mode and not api_key:
            st.warning("⚠️ Please provide an LLM API Key in the sidebar or toggle Demo Mode.")
        else:
            # Append User message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant response
            with st.chat_message("assistant"):
                status_placeholder = st.status("Executing Multi-Agent Graph...", expanded=True)
                with status_placeholder:
                    st.write("🔍 **Retriever Agent**: Querying vector database & ranking chunks...")
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    try:
                        st.write("🧠 **Analyst Agent**: Running tools, verifying evidence & calculating...")
                        final_ans, final_state = execute_query(
                            question=user_query,
                            history=history_payload,
                            max_rounds=max_rounds,
                            use_demo_mode=demo_mode,
                            api_key=api_key,
                            model_name=model_name,
                            base_url=base_url,
                            temperature=temperature,
                        )
                        st.write("✍️ **Answer Agent**: Formatting grounded response with verified citations...")
                        status_placeholder.update(label="Complete!", state="complete", expanded=False)

                    except Exception as exc:
                        status_placeholder.update(label="Failed", state="error", expanded=False)
                        st.error(f"Execution failed: {exc}")
                        logger.exception("Chat failed")
                        final_ans = None
                        final_state = None

                if final_ans:
                    st.markdown(final_ans.text)

                    # Build Trace Data for storage
                    trace_data = {
                        "rounds_used": final_ans.rounds_used,
                        "citations": final_ans.citations,
                        "analysis": final_state.get("analysis") if final_state else None,
                        "chunks_by_id": final_state.get("chunks_by_id", {}) if final_state else {},
                        "attempted_queries": final_state.get("attempted_queries", []) if final_state else [],
                        "rewritten_queries": final_state.get("rewritten_queries", []) if final_state else [],
                    }

                    # Show Citations below
                    if final_ans.citations:
                        with st.expander(f"📚 View {len(final_ans.citations)} Citations", expanded=False):
                            for cit in final_ans.citations:
                                page_str = f" • Page {cit.page}" if cit.page else ""
                                chunk = (final_state.get("chunks_by_id", {}) or {}).get(cit.chunk_id)
                                st.markdown(f"**{cit.doc_name}**{page_str} (`{cit.chunk_id}`)")
                                if chunk:
                                    st.caption(f"> {chunk.text[:250]}...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_ans.text,
                        "trace": trace_data,
                    })

    if st.session_state.messages:
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ── Tab 2: Evidence Explorer ──────────────────────────────────────────────────
with tab_explorer:
    st.subheader("📚 Chroma Vector Database Inspector")
    search_term = st.text_input("Filter indexed chunks by keyword:", placeholder="e.g. ConvNeXt, 2025, accuracy...")

    try:
        stored = db.vector_store.get(include=["documents", "metadatas"])
        contents = stored.get("documents") or []
        metas = stored.get("metadatas") or []
        ids = stored.get("ids") or []

        matched = []
        for cid, content, meta in zip(ids, contents, metas):
            if not search_term or (search_term.lower() in content.lower()) or (search_term.lower() in str(meta).lower()):
                matched.append({"id": cid, "content": content, "metadata": meta or {}})

        st.caption(f"Showing {len(matched)} of {len(contents)} chunks in database.")

        for item in matched[:50]:
            meta = item["metadata"]
            doc_name = meta.get("doc_name") or meta.get("filename") or "Document"
            page = f"Page {meta.get('page')}" if meta.get("page") else "No Page"
            ctype = meta.get("chunk_type", "text")

            with st.expander(f"📄 {doc_name} • {page} • [{ctype}] • ID: {item['id']}"):
                st.code(item["content"], language="markdown" if ctype == "table" else None)
                st.json(meta)

    except Exception as exc:
        st.error(f"Failed to inspect vector database: {exc}")

# ── Tab 3: Analyst Tool Sandbox ───────────────────────────────────────────────
with tab_tools:
    st.subheader("🧪 Analyst Python Tools Sandbox")
    st.markdown("Test the deterministic tools available to the Analyst Agent directly:")

    tool_choice = st.selectbox(
        "Select Tool",
        ["Calculator (Arithmetic)", "Table Extractor", "Metric Analysis (Ranking & Trend)"],
    )

    if tool_choice == "Calculator (Arithmetic)":
        from app.agents.analyst.tools.calculator import calculate, CalculatorError

        expr = st.text_input("Expression", value="(2850 - 1200) / 1200 * 100")
        desc = st.text_input("Description", value="ViT vs ResNet training cost increase (%)")
        if st.button("Compute Result"):
            try:
                calc_res = calculate(expr, desc)
                st.success(f"**Result**: {calc_res.result}")
                st.json(calc_res.model_dump())
            except CalculatorError as err:
                st.error(f"Calculator Error: {err}")

    elif tool_choice == "Table Extractor":
        from app.agents.analyst.tools.table_extractor import extract_tables, to_records

        sample_table = st.text_area(
            "Markdown Table",
            value=(
                "| Model | Params (M) | Top-1 Accuracy | Latency (ms) |\n"
                "| :--- | :---: | :---: | :---: |\n"
                "| ResNet-50 | 25.6 | 76.1% | 12.4 |\n"
                "| ViT-B/16 | 86.6 | 81.8% | 29.5 |\n"
                "| ConvNeXt-T | 28.6 | 82.1% | 14.2 |"
            ),
            height=150,
        )
        if st.button("Extract Tables"):
            tables = extract_tables(sample_table, chunk_id="test_chunk")
            if tables:
                for t in tables:
                    records = to_records(t)
                    st.dataframe(records)
                    st.json(t.model_dump())
            else:
                st.warning("No table detected.")

    elif tool_choice == "Metric Analysis (Ranking & Trend)":
        from app.agents.analyst.tools.data_analysis import analyze_numbers, analyze_trend

        analysis_mode = st.radio("Mode", ["Ranking", "Trend"], horizontal=True)
        if analysis_mode == "Ranking":
            nums_str = st.text_input("Values (comma-separated)", "76.1, 81.8, 82.1, 81.3")
            labels_str = st.text_input("Labels (comma-separated)", "ResNet-50, ViT-B/16, ConvNeXt-T, Swin-T")
            metric = st.text_input("Metric Name", "Top-1 Accuracy")
            if st.button("Analyze Ranking"):
                try:
                    vals = [float(x.strip()) for x in nums_str.split(",") if x.strip()]
                    labs = [x.strip() for x in labels_str.split(",") if x.strip()]
                    res = analyze_numbers(vals, metric, labels=labs)
                    st.success(res.summary)
                    st.json(res.details)
                except Exception as e:
                    st.error(str(e))
        else:
            trend_vals = st.text_input("Values", "78.4, 80.1, 81.5, 82.9")
            periods_str = st.text_input("Periods", "2022, 2023, 2024, 2025")
            metric_t = st.text_input("Metric", "Top-1 Accuracy")
            if st.button("Analyze Trend"):
                try:
                    vals = [float(x.strip()) for x in trend_vals.split(",") if x.strip()]
                    pers = [x.strip() for x in periods_str.split(",") if x.strip()]
                    res = analyze_trend(vals, pers, metric_t)
                    st.success(res.summary)
                    st.json(res.details)
                except Exception as e:
                    st.error(str(e))