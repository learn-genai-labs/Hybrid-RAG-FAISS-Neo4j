import sys
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------------
# Project path setup
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.ingestion.ingestion_pipeline import run_ingestion_pipeline
from backend.retrieval.qa_pipeline import answer_question


# ------------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------------

UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔗",
    layout="wide",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "ingestion_complete" not in st.session_state:
    st.session_state.ingestion_complete = False

if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None

if "qa_result" not in st.session_state:
    st.session_state.qa_result = None


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("Hybrid RAG")

st.caption(
    "Vector Search + Knowledge Graph + Guardrails + Evaluation"
)

st.markdown(
    """
Upload a PDF to build two knowledge representations:

- **FAISS** for semantic vector retrieval
- **Neo4j** for entity and relationship retrieval

Questions are answered using evidence retrieved from both systems.
"""
)

st.divider()


# ==================================================================
# PIPELINE 1
# ==================================================================

st.header("1. Document Ingestion")

st.write(
    "Upload a PDF to create the FAISS vector index "
    "and Neo4j Knowledge Graph."
)


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    st.info(
        f"Selected document: **{uploaded_file.name}**"
    )

    if st.button(
        "Ingest Document",
        type="primary",
        use_container_width=True,
    ):

        destination = (
            UPLOAD_DIRECTORY
            / uploaded_file.name
        )

        # ----------------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------------

        try:

            with open(
                destination,
                "wb",
            ) as output_file:

                output_file.write(
                    uploaded_file.getbuffer()
                )

        except Exception as error:

            st.error(
                f"Unable to save uploaded PDF: {error}"
            )

            st.stop()

        # ----------------------------------------------------------
        # Run Pipeline 1
        # ----------------------------------------------------------

        try:

            with st.status(
                "Running Hybrid RAG ingestion...",
                expanded=True,
            ) as status:

                st.write(
                    "Loading PDF..."
                )

                st.write(
                    "Chunking document..."
                )

                st.write(
                    "Creating embeddings and "
                    "FAISS vector index..."
                )

                st.write(
                    "Extracting entities and "
                    "relationships..."
                )

                st.write(
                    "Building Neo4j Knowledge Graph..."
                )

                ingestion_result = (
                    run_ingestion_pipeline(
                        str(destination),
                        clear_existing_graph=True,
                    )
                )

                status.update(
                    label="Ingestion completed",
                    state="complete",
                    expanded=False,
                )

            st.session_state.ingestion_complete = True

            st.session_state.ingested_file = (
                uploaded_file.name
            )

            st.session_state.ingestion_result = (
                ingestion_result
            )

            # New document means previous answer is obsolete.
            st.session_state.qa_result = None

        except Exception as error:

            st.session_state.ingestion_complete = False

            st.error(
                "Document ingestion failed."
            )

            st.exception(
                error
            )


# ------------------------------------------------------------------
# Ingestion status
# ------------------------------------------------------------------

if st.session_state.ingestion_complete:

    result = st.session_state.ingestion_result

    st.success(
        "Document successfully ingested into "
        "FAISS and Neo4j."
    )

    st.write(
        f"**Active document:** "
        f"{st.session_state.ingested_file}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "PDF Pages",
            result.get(
                "pages",
                0,
            ),
        )

    with col2:
        st.metric(
            "Chunks",
            result.get(
                "chunks",
                0,
            ),
        )

    with col3:
        st.metric(
            "KG Nodes",
            result.get(
                "neo4j_nodes",
                0,
            ),
        )

    with col4:
        st.metric(
            "KG Relationships",
            result.get(
                "neo4j_relationships",
                0,
            ),
        )

    vector_col, graph_col = st.columns(2)

    with vector_col:

        st.success(
            "✓ FAISS Vector Database Ready"
        )

    with graph_col:

        st.success(
            "✓ Neo4j Knowledge Graph Ready"
        )


st.divider()


# ==================================================================
# PIPELINE 2
# ==================================================================

st.header("2. Hybrid Question & Answer")

if not st.session_state.ingestion_complete:

    st.warning(
        "Upload and ingest a PDF before asking questions."
    )

else:

    question = st.text_area(
        "Ask a question about the document",
        placeholder=(
            "Example: What databases and storage "
            "technologies are used in the architecture?"
        ),
        height=100,
    )

    if st.button(
        "Ask Hybrid RAG",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            try:

                with st.spinner(
                    "Searching FAISS and Neo4j..."
                ):

                    qa_result = answer_question(
                        question=question,
                        run_evaluation=True,
                    )

                st.session_state.qa_result = (
                    qa_result
                )

            except Exception as error:

                st.error(
                    "Question answering failed."
                )

                st.exception(
                    error
                )


# ==================================================================
# Q&A RESULT
# ==================================================================

result = st.session_state.qa_result


if result:

    st.divider()

    # --------------------------------------------------------------
    # Guardrail result
    # --------------------------------------------------------------

    if result.get("status") == "blocked":

        st.error(
            "Request blocked by guardrail."
        )

        st.write(
            f"**Stage:** "
            f"{result.get('stage')}"
        )

        st.write(
            f"**Reason:** "
            f"{result.get('reason')}"
        )

        if result.get("answer"):

            st.write(
                result["answer"]
            )

    else:

        # ----------------------------------------------------------
        # Answer
        # ----------------------------------------------------------

        st.subheader("Answer")

        st.markdown(
            result["answer"]
        )

        st.success(
            "✓ Guardrails passed"
        )

        # ----------------------------------------------------------
        # Sources
        # ----------------------------------------------------------

        st.subheader("Sources")

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            for source in sources:

                st.write(
                    f"• **{source.get('file_name')}** "
                    f"— Page {source.get('page')}, "
                    f"Chunk {source.get('chunk_id')}"
                )

        else:

            st.info(
                "No source metadata available."
            )

        # ----------------------------------------------------------
        # Retrieval summary
        # ----------------------------------------------------------

        vector_results = result.get(
            "vector_results",
            [],
        )

        graph_results = result.get(
            "graph_results",
            [],
        )

        st.subheader(
            "Hybrid Retrieval"
        )

        vector_col, graph_col = st.columns(2)

        with vector_col:

            st.metric(
                "FAISS Results",
                len(vector_results),
            )

        with graph_col:

            st.metric(
                "Neo4j Relationships",
                len(graph_results),
            )

        # ----------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------

        evaluation = result.get(
            "evaluation"
        )

        if evaluation:

            st.subheader(
                "RAG Evaluation"
            )

            if (
                evaluation.get("status")
                == "evaluation_failed"
            ):

                st.warning(
                    "Evaluation could not be completed."
                )

                st.write(
                    evaluation.get(
                        "error"
                    )
                )

            else:

                retrieval_metric = evaluation[
                    "retrieval_relevance"
                ]

                faithfulness_metric = evaluation[
                    "faithfulness"
                ]

                relevance_metric = evaluation[
                    "answer_relevance"
                ]

                eval_col1, eval_col2, eval_col3 = (
                    st.columns(3)
                )

                with eval_col1:

                    st.metric(
                        "Retrieval Relevance",
                        f"{retrieval_metric['score']:.2f}",
                    )

                with eval_col2:

                    st.metric(
                        "Faithfulness",
                        f"{faithfulness_metric['score']:.2f}",
                    )

                with eval_col3:

                    st.metric(
                        "Answer Relevance",
                        f"{relevance_metric['score']:.2f}",
                    )

                st.write(
                    f"**Average evaluation score:** "
                    f"{evaluation['average_score']:.3f}"
                )

                with st.expander(
                    "View evaluation reasoning"
                ):

                    st.markdown(
                        "#### Retrieval Relevance"
                    )

                    st.write(
                        retrieval_metric[
                            "reason"
                        ]
                    )

                    st.markdown(
                        "#### Faithfulness"
                    )

                    st.write(
                        faithfulness_metric[
                            "reason"
                        ]
                    )

                    st.markdown(
                        "#### Answer Relevance"
                    )

                    st.write(
                        relevance_metric[
                            "reason"
                        ]
                    )

        # ----------------------------------------------------------
        # Technical retrieval evidence
        # ----------------------------------------------------------

        with st.expander(
            "View retrieved evidence"
        ):

            st.markdown(
                "### FAISS Vector Results"
            )

            for index, item in enumerate(
                vector_results,
                start=1,
            ):

                metadata = item.get(
                    "metadata",
                    {},
                )

                st.markdown(
                    f"**Vector Result {index}**"
                )

                st.caption(
                    f"Page {metadata.get('page')} | "
                    f"Chunk {metadata.get('chunk_id')} | "
                    f"Distance: "
                    f"{item.get('score', 0):.4f}"
                )

                st.write(
                    item.get(
                        "content",
                        "",
                    )
                )

                st.divider()

            st.markdown(
                "### Neo4j Graph Results"
            )

            if graph_results:

                for index, item in enumerate(
                    graph_results,
                    start=1,
                ):

                    st.write(
                        f"**{index}. "
                        f"{item.get('source')} "
                        f"—[{item.get('relationship')}]→ "
                        f"{item.get('target')}**"
                    )

            else:

                st.write(
                    "No matching graph relationships."
                )


# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------

st.divider()

with st.expander(
    "Hybrid RAG Architecture"
):

    st.code(
        """
PDF Upload
    |
    v
PDF Loader
    |
    v
Text Chunking
    |
    +--------------------------+
    |                          |
    v                          v
OpenAI Embeddings       Entity / Relationship
    |                      Extraction
    v                          |
  FAISS                        v
                         Neo4j Graph
    |                          |
    +-------------+------------+
                  |
                  v
             User Question
                  |
                  v
           Input Guardrail
                  |
        +---------+---------+
        |                   |
        v                   v
   FAISS Search        Neo4j Search
        |                   |
        +---------+---------+
                  |
                  v
           Hybrid Context
                  |
                  v
          Context Guardrail
                  |
                  v
             OpenAI LLM
                  |
                  v
          Output Guardrail
                  |
                  v
                Answer
                  |
                  v
                 Evals
        +---------+---------+
        |         |         |
        v         v         v
    Retrieval  Faithful   Answer
    Relevance    ness     Relevance
        """,
        language="text",
    )