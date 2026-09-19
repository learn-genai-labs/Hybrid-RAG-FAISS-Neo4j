from typing import Any

from backend.graph.neo4j_client import Neo4jClient
from backend.ingestion.graph_extractor import (
    extract_graph_from_chunk,
)
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_chunker import chunk_documents


def normalize_relationship_type(
    relationship_type: str,
) -> str:
    """
    Convert a relationship type into a safe Neo4j
    relationship name.
    """

    normalized = (
        relationship_type
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if not normalized:
        return "RELATED_TO"

    return normalized


def ingest_graph_data(
    graph_data: dict[str, Any],
    client: Neo4jClient,
) -> None:
    """
    Store extracted entities and relationships in Neo4j.
    """

    entities = graph_data.get(
        "entities",
        [],
    )

    relationships = graph_data.get(
        "relationships",
        [],
    )

    metadata = graph_data.get(
        "metadata",
        {},
    )

    source_file = metadata.get(
        "file_name",
        "unknown",
    )

    page = metadata.get(
        "page",
    )

    chunk_id = metadata.get(
        "chunk_id",
    )

    # ---------------------------------------------------------
    # Collect all entities
    # ---------------------------------------------------------

    entity_map = {}

    for entity in entities:

        name = entity.get(
            "name",
            "",
        ).strip()

        entity_type = entity.get(
            "type",
            "Concept",
        ).strip()

        if name:
            entity_map[name] = entity_type

    # ---------------------------------------------------------
    # Ensure relationship endpoints also exist as entities
    # ---------------------------------------------------------

    for relationship in relationships:

        source = relationship.get(
            "source",
            "",
        ).strip()

        target = relationship.get(
            "target",
            "",
        ).strip()

        if source and source not in entity_map:
            entity_map[source] = "Concept"

        if target and target not in entity_map:
            entity_map[target] = "Concept"

    # ---------------------------------------------------------
    # Create / merge entity nodes
    # ---------------------------------------------------------

    with client.driver.session() as session:

        for name, entity_type in entity_map.items():

            session.run(
                """
                MERGE (entity:Entity {name: $name})

                ON CREATE SET
                    entity.type = $entity_type,
                    entity.source_file = $source_file,
                    entity.page = $page,
                    entity.chunk_id = $chunk_id

                ON MATCH SET
                    entity.type =
                        CASE
                            WHEN entity.type IS NULL
                            THEN $entity_type
                            ELSE entity.type
                        END
                """,
                name=name,
                entity_type=entity_type,
                source_file=source_file,
                page=page,
                chunk_id=chunk_id,
            )

        # -----------------------------------------------------
        # Create relationships
        # -----------------------------------------------------

        for relationship in relationships:

            source = relationship.get(
                "source",
                "",
            ).strip()

            target = relationship.get(
                "target",
                "",
            ).strip()

            relationship_type = (
                relationship.get(
                    "relationship",
                    "RELATED_TO",
                )
            )

            if not source or not target:
                continue

            relationship_type = (
                normalize_relationship_type(
                    relationship_type
                )
            )

            query = f"""
            MATCH (source:Entity {{name: $source}})
            MATCH (target:Entity {{name: $target}})

            MERGE (source)-[relationship:{relationship_type}]->(target)

            ON CREATE SET
                relationship.source_file = $source_file,
                relationship.page = $page,
                relationship.chunk_id = $chunk_id
            """

            session.run(
                query,
                source=source,
                target=target,
                source_file=source_file,
                page=page,
                chunk_id=chunk_id,
            )


def clear_knowledge_graph(
    client: Neo4jClient,
) -> None:
    """
    Delete all nodes and relationships.

    Used only during local development/testing.
    """

    with client.driver.session() as session:

        session.run(
            "MATCH (node) DETACH DELETE node"
        )


if __name__ == "__main__":

    TEST_PDF = (
        "data/uploads/"
        "spotify_web_app_architecture.pdf"
    )

    print("\n" + "=" * 70)
    print("NEO4J KNOWLEDGE GRAPH INGESTION TEST")
    print("=" * 70)

    client = Neo4jClient()

    try:

        # -----------------------------------------------------
        # Step 1 - Verify Neo4j
        # -----------------------------------------------------

        print("\n1. Connecting to Neo4j...")

        client.verify_connection()

        # -----------------------------------------------------
        # Step 2 - Clear previous test graph
        # -----------------------------------------------------

        print("\n2. Clearing previous test graph...")

        clear_knowledge_graph(
            client
        )

        print("   Previous graph cleared.")

        # -----------------------------------------------------
        # Step 3 - Load PDF
        # -----------------------------------------------------

        print("\n3. Loading PDF...")

        documents = load_pdf(
            TEST_PDF
        )

        print(
            f"   Pages loaded: {len(documents)}"
        )

        # -----------------------------------------------------
        # Step 4 - Chunk PDF
        # -----------------------------------------------------

        print("\n4. Chunking PDF...")

        chunks = chunk_documents(
            documents
        )

        print(
            f"   Chunks available: {len(chunks)}"
        )

        # -----------------------------------------------------
        # Step 5 - Use only Chunk 1
        # -----------------------------------------------------

        test_chunk = chunks[0]

        print(
            "\n5. Extracting graph from "
            "Chunk 1..."
        )

        graph_data = extract_graph_from_chunk(
            test_chunk
        )

        print(
            f"   Entities extracted: "
            f"{len(graph_data['entities'])}"
        )

        print(
            f"   Relationships extracted: "
            f"{len(graph_data['relationships'])}"
        )

        # -----------------------------------------------------
        # Step 6 - Insert into Neo4j
        # -----------------------------------------------------

        print(
            "\n6. Inserting graph into Neo4j..."
        )

        ingest_graph_data(
            graph_data,
            client,
        )

        print(
            "   Knowledge Graph inserted successfully."
        )

        # -----------------------------------------------------
        # Step 7 - Verify counts
        # -----------------------------------------------------

        with client.driver.session() as session:

            node_result = session.run(
                """
                MATCH (node)
                RETURN count(node) AS count
                """
            ).single()

            relationship_result = session.run(
                """
                MATCH ()-[relationship]->()
                RETURN count(relationship) AS count
                """
            ).single()

            node_count = node_result["count"]

            relationship_count = (
                relationship_result["count"]
            )

        print("\n" + "=" * 70)
        print("NEO4J RESULT")
        print("=" * 70)

        print(
            f"\nNodes created: {node_count}"
        )

        print(
            f"Relationships created: "
            f"{relationship_count}"
        )

        print("\n" + "=" * 70)
        print(
            "KNOWLEDGE GRAPH INGESTION "
            "TEST COMPLETED"
        )
        print("=" * 70)

    finally:

        client.close()