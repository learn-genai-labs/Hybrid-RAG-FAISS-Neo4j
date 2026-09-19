from backend.graph.neo4j_client import Neo4jClient


def search_knowledge_graph(
    user_question: str,
    limit: int = 10,
) -> list[dict]:
    """
    Search Neo4j for entities related to words
    appearing in the user's question.

    Returns matching graph relationships.
    """

    if not user_question.strip():
        raise ValueError(
            "Graph search query cannot be empty."
        )

    client = Neo4jClient()

    try:
        client.verify_connection()

        cypher_query = """
        MATCH (source:Entity)-[relationship]->(target:Entity)

        WHERE
            any(
                word IN split(toLower($user_question), ' ')
                WHERE size(word) >= 3
                AND (
                    toLower(source.name) CONTAINS word
                    OR toLower(target.name) CONTAINS word
                )
            )

        RETURN
            source.name AS source,
            source.type AS source_type,
            type(relationship) AS relationship,
            target.name AS target,
            target.type AS target_type,
            relationship.source_file AS source_file,
            relationship.page AS page,
            relationship.chunk_id AS chunk_id

        LIMIT $limit
        """

        with client.driver.session() as session:

            result = session.run(
                cypher_query,
                user_question=user_question,
                limit=limit,
            )

            records = [
                dict(record)
                for record in result
            ]

        return records

    finally:
        client.close()


if __name__ == "__main__":

    question = (
        "What databases and storage technologies "
        "are used?"
    )

    print("\n" + "=" * 70)
    print("NEO4J GRAPH RETRIEVAL TEST")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    results = search_knowledge_graph(
        user_question=question,
        limit=10,
    )

    print(
        f"\nGraph results: {len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRESULT {index}"
        )

        print(
            f"{result['source']} "
            f"--[{result['relationship']}]--> "
            f"{result['target']}"
        )

        print(
            f"Source: {result['source_file']} "
            f"| Page: {result['page']} "
            f"| Chunk: {result['chunk_id']}"
        )