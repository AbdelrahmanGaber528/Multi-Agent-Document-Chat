import os
from typing import Any

import httpx
import streamlit as st

# ── Page Configuration & Styling ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Document Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stMetric {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    .agent-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        background-color: #1e6347;
        color: #c6f6d5;
    }
    .citation-card {
        border-left: 3px solid #3182ce;
        background-color: rgba(49, 130, 206, 0.08);
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── API Client Helpers ────────────────────────────────────────────────────────
def _extract_error(response: httpx.Response) -> str:
    """Parse FastAPI HTTPException or AppError JSON payloads cleanly."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            if "detail" in payload:
                return str(payload["detail"])
            if "message" in payload:
                return str(payload["message"])
    except Exception:
        pass
    return response.text.strip() or f"HTTP {response.status_code}"


def api_health(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url}/api/health", timeout=3.0)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except httpx.HTTPError:
        return False


def api_list_resources(base_url: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        resp = httpx.get(f"{base_url}/resources", timeout=15.0)
        if resp.status_code != 200:
            return [], _extract_error(resp)
        data = resp.json()
        return data.get("resources", []), None
    except httpx.HTTPError as exc:
        return [], f"Cannot reach backend: {exc}"


def api_upload_resource(
    base_url: str, filename: str, content: bytes, mime_type: str | None
) -> dict[str, Any]:
    files = {"file": (filename, content, mime_type or "application/octet-stream")}
    resp = httpx.post(f"{base_url}/resources/upload", files=files, timeout=120.0)
    if resp.status_code != 200:
        raise RuntimeError(_extract_error(resp))
    return resp.json()


def api_delete_resource(base_url: str, resource_id: str) -> None:
    resp = httpx.delete(f"{base_url}/resources/{resource_id}", timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(_extract_error(resp))


def api_chat(
    base_url: str, question: str, history: list[dict[str, str]]
) -> dict[str, Any]:
    payload = {"question": question, "history": history}
    # Multi-round retrieval + tool loops can take up to ~60-90s
    resp = httpx.post(f"{base_url}/chat", json=payload, timeout=120.0)
    if resp.status_code != 200:
        raise RuntimeError(_extract_error(resp))
    return resp.json()


# ── UI Rendering Helpers ──────────────────────────────────────────────────────
def render_assistant_metadata(meta: dict[str, Any]) -> None:
    rounds = meta.get("rounds_used", 1)
    citations = meta.get("citations") or []
    insufficient = meta.get("insufficient_evidence", False)
    missing_info = meta.get("missing_info") or []

    st.markdown(
        f"<span class='agent-badge'>Rounds Used: {rounds}</span>"
        f"<span class='agent-badge'>Verified Citations: {len(citations)}</span>",
        unsafe_allow_html=True,
    )

    if insufficient:
        st.warning(
            "Some requested information could not be fully verified in the uploaded documents."
        )
        if missing_info:
            with st.expander("Missing Information Details", expanded=False):
                for item in missing_info:
                    st.markdown(f"- {item}")

    if citations:
        with st.expander(f"📚 View {len(citations)} Verified Citations", expanded=False):
            for cit in citations:
                doc_name = cit.get("doc_name") or "Unknown Document"
                page = cit.get("page")
                section = cit.get("section")
                chunk_id = cit.get("chunk_id", "")

                page_str = f" • Page {page}" if page is not None else ""
                section_str = f" • <i>{section}</i>" if section else ""
                st.markdown(
                    f"<div class='citation-card'><b>{doc_name}</b>{page_str}{section_str} "
                    f"(<code>{chunk_id}</code>)</div>",
                    unsafe_allow_html=True,
                )


# ── Sidebar: Backend Connection & Knowledge Base ──────────────────────────────
with st.sidebar:
    st.title("⚙️ System Control")

    api_url = st.text_input(
        "Backend API URL",
        value=DEFAULT_API_URL,
        help="FastAPI server address (e.g. http://localhost:8000)",
    ).rstrip("/")

    backend_online = api_health(api_url)
    if backend_online:
        st.success("● Backend Online")
    else:
        st.error("● Backend Offline — start FastAPI (`docker compose up`)")

    st.divider()
    st.subheader("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        disabled=not backend_online,
    )

    if uploaded_files and st.button(
        "📥 Process & Ingest Files", use_container_width=True, disabled=not backend_online
    ):
        success_count = 0
        with st.spinner("Uploading and indexing documents..."):
            for uf in uploaded_files:
                try:
                    res = api_upload_resource(
                        api_url, uf.name, uf.getvalue(), uf.type
                    )
                    num_chunks = (
                        res.get("num_chunks")
                        or res.get("chunk_count")
                        or res.get("chunks")
                    )
                    suffix = f" ({num_chunks} chunks)" if num_chunks is not None else ""
                    st.success(f"✓ {uf.name}{suffix}")
                    success_count += 1
                except Exception as exc:
                    st.error(f"Error ingesting {uf.name}: {exc}")
        if success_count > 0:
            st.rerun()

    current_docs, list_err = (
        api_list_resources(api_url) if backend_online else ([], None)
    )
    if list_err:
        st.warning(list_err)

    total_chunks = sum(int(d.get("chunks", 0)) for d in current_docs)
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(current_docs))
    col2.metric("Total Chunks", total_chunks)

    if current_docs:
        with st.expander("📄 Document Inventory", expanded=True):
            for doc in current_docs:
                doc_id = doc["doc_id"]
                filename = doc.get("filename") or "Unknown Document"
                chunks = doc.get("chunks", 0)

                dcol1, dcol2 = st.columns([4, 1])
                dcol1.markdown(f"**{filename}**  \n`{chunks} chunks`")
                if dcol2.button("🗑️", key=f"del_{doc_id}", help="Delete document"):
                    try:
                        api_delete_resource(api_url, doc_id)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

            if st.button(
                "Clear All Documents", type="secondary", use_container_width=True
            ):
                for doc in current_docs:
                    try:
                        api_delete_resource(api_url, doc["doc_id"])
                    except Exception as exc:
                        st.error(f"Failed deleting {doc.get('filename')}: {exc}")
                st.rerun()

    st.divider()
    with st.expander("ℹ️ Multi-Agent Architecture", expanded=False):
        st.markdown(
            """
            - **🔍 Retriever Agent**: Hybrid semantic + keyword search with Reciprocal Rank Fusion.
            - **🧠 Analyst Agent**: Fact verification using safe Python arithmetic, table extraction, trends, and cross-doc comparison.
            - **🔄 Feedback Loop**: Detects missing facts and re-queries the Retriever automatically.
            - **✍️ Answer Agent**: Synthesizes grounded responses with verified inline citations.
            """
        )


# ── Main Chat Interface ───────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Document Chat")
st.markdown(
    "Ask complex multi-document questions with automated **table extraction**, "
    "**quantitative calculation**, and **strict source verification**."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            render_assistant_metadata(msg["meta"])

user_query = st.chat_input(
    "Ask a question about the uploaded documents...",
    disabled=not backend_online,
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Running multi-agent retrieval, analysis, and verification..."):
            try:
                response_data = api_chat(api_url, user_query, history_payload)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                response_data = None

        if response_data:
            answer_text = response_data.get("answer", "")
            st.markdown(answer_text)

            meta = {
                "citations": response_data.get("citations", []),
                "insufficient_evidence": response_data.get("insufficient_evidence", False),
                "missing_info": response_data.get("missing_info", []),
                "rounds_used": response_data.get("rounds_used", 1),
            }
            render_assistant_metadata(meta)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "meta": meta,
                }
            )

if st.session_state.messages:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()