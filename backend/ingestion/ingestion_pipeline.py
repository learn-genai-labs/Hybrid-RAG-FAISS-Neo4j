from pathlib import Path

from backend.graph.neo4j_client import Neo4jClient
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_chunker import chunk_documents
from backend.ingestion.vector_ingestion import (
    create_faiss_vector_store,
    save_faiss_vector_store,
)
from backend.ingestion.graph_extractor import (
    extract_graph_from_chunk,
)
from backend.ingestion.graph_ingestion import (
    clear_knowledge_graph,
    ingest_graph_data,
)


def run_ingestion_pipeline(
    pdf_path: str,
    clear_existing_graph: bool = True,
) -> dict:
    """
    Complete Hybrid RAG ingestion pipeline.

    PDF
      -> Load
      -> Chunk
      -> Embeddings
      -> FAISS
      -> Knowledge Graph extraction
      -> Neo4j
    """

    file_path = Path(pdf_path)

    print("\n" + "=" * 70)
    print("HYBRID RAG - INGESTION PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load PDF
    # ---------------------------------------------------------

    print("\n[1/6] Loading PDF...")

    documents = load_pdf(
        str(file_path)
    )

    print(
        f"      Pages loaded: {len(documents)}"
    )

    # ---------------------------------------------------------
    # 2. Chunk documents
    # ---------------------------------------------------------

    print("\n[2/6] Chunking documents...")

    chunks = chunk_documents(
        documents
    )

    print(
        f"      Chunks created: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # 3. Create FAISS
    # ---------------------------------------------------------

    print("\n[3/6] Creating FAISS vector store...")

    vector_store = create_faiss_vector_store(
        chunks
    )

    save_faiss_vector_store(
        vector_store
    )

    print(
        "      FAISS ingestion completed."
    )

    # ---------------------------------------------------------
    # 4. Connect to Neo4j
    # ---------------------------------------------------------

    print("\n[4/6] Connecting to Neo4j...")

    neo4j_client = Neo4jClient()

    neo4j_client.verify_connection()

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    total_entities = 0
    total_relationships = 0

    try:

        if clear_existing_graph:

            print(
                "      Clearing existing graph..."
            )

            clear_knowledge_graph(
                neo4j_client
            )

        # -----------------------------------------------------
        # 5. Process every chunk for KG
        # -----------------------------------------------------

        print(
            "\n[5/6] Building Knowledge Graph..."
        )

        total_chunks = len(chunks)

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"      Processing chunk "
                f"{index}/{total_chunks}..."
            )

            try:

                graph_data = (
                    extract_graph_from_chunk(
                        chunk
                    )
                )

                entity_count = len(
                    graph_data.get(
                        "entities",
                        [],
                    )
                )

                relationship_count = len(
                    graph_data.get(
                        "relationships",
                        [],
                    )
                )

                total_entities += (
                    entity_count
                )

                total_relationships += (
                    relationship_count
                )

                ingest_graph_data(
                    graph_data,
                    neo4j_client,
                )

                print(
                    f"         Entities: "
                    f"{entity_count}, "
                    f"Relationships: "
                    f"{relationship_count}"
                )

            except Exception as error:

                print(
                    f"         WARNING: "
                    f"Chunk {index} failed."
                )

                print(
                    f"         Error: {error}"
                )

        # -----------------------------------------------------
        # 6. Get final Neo4j counts
        # -----------------------------------------------------

        print(
            "\n[6/6] Verifying Neo4j graph..."
        )

        with neo4j_client.driver.session() as session:

            node_record = session.run(
                """
                MATCH (n)
                RETURN count(n) AS count
                """
            ).single()

            relationship_record = session.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS count
                """
            ).single()

            neo4j_nodes = (
                node_record["count"]
            )

            neo4j_relationships = (
                relationship_record["count"]
            )

    finally:

        neo4j_client.close()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    result = {
        "pdf": file_path.name,
        "pages": len(documents),
        "chunks": len(chunks),
        "extracted_entities": total_entities,
        "extracted_relationships": total_relationships,
        "neo4j_nodes": neo4j_nodes,
        "neo4j_relationships": (
            neo4j_relationships
        ),
    }

    print("\n" + "=" * 70)
    print("INGESTION PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nPDF: {result['pdf']}"
    )

    print(
        f"Pages: {result['pages']}"
    )

    print(
        f"Chunks: {result['chunks']}"
    )

    print(
        f"Extracted entities: "
        f"{result['extracted_entities']}"
    )

    print(
        f"Extracted relationships: "
        f"{result['extracted_relationships']}"
    )

    print(
        f"Neo4j nodes: "
        f"{result['neo4j_nodes']}"
    )

    print(
        f"Neo4j relationships: "
        f"{result['neo4j_relationships']}"
    )

    print("\nFAISS: READY")
    print("NEO4J: READY")

    print("=" * 70)

    return result


# -------------------------------------------------------------
# Command-line execution
# -------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python -m "
            "backend.ingestion.ingestion_pipeline "
            "\"data\\uploads\\your_file.pdf\"\n"
        )

        raise SystemExit(1)

    run_ingestion_pipeline(
        sys.argv[1]
    )