import json
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from backend.config.settings import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_chunker import chunk_documents


def create_graph_extraction_prompt(text: str) -> str:
    """
    Create a prompt for extracting knowledge graph
    entities and relationships from document text.
    """

    return f"""
You are a knowledge graph extraction system.

Extract important entities and relationships ONLY from the
provided text.

Do not invent information that is not present in the text.

Return valid JSON only.

Use this exact structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "relationship": "RELATIONSHIP_TYPE",
      "target": "target entity name"
    }}
  ]
}}

Rules:

1. Entity names must be concise and meaningful.

2. Entity types should use simple categories such as:
   Technology,
   Service,
   Database,
   Storage,
   Platform,
   Component,
   Data,
   Feature,
   Tool,
   Concept.

3. Relationship names must be uppercase and use underscores.

Examples:

STORES
USES
DEPENDS_ON
CACHES
INDEXES
RUNS_ON
COMMUNICATES_WITH
SENDS_DATA_TO
USES_DATA_FROM

4. Every source and target appearing in a relationship must
   also appear in the entities list.

5. Extract only relationships supported by the supplied text.

6. Do not infer unsupported relationships.

7. Do not include Markdown.

8. Do not include explanations outside the JSON.

TEXT:

{text}
""".strip()


def extract_graph_from_text(
    text: str,
) -> dict[str, Any]:
    """
    Extract entities and relationships from text using
    the configured OpenAI chat model.
    """

    if not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured in .env."
        )

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    prompt = create_graph_extraction_prompt(text)

    response = llm.invoke(prompt)

    response_text = response.content

    if not isinstance(response_text, str):
        raise ValueError(
            "Unexpected response format received from LLM."
        )

    cleaned_response = response_text.strip()

    # ---------------------------------------------------------
    # Remove Markdown fences if returned by the model
    # ---------------------------------------------------------

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]

    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    cleaned_response = cleaned_response.strip()

    # ---------------------------------------------------------
    # Convert JSON string to Python dictionary
    # ---------------------------------------------------------

    try:
        graph_data = json.loads(cleaned_response)

    except json.JSONDecodeError as error:
        raise ValueError(
            "LLM did not return valid JSON.\n\n"
            f"Raw response:\n{response_text}"
        ) from error

    # ---------------------------------------------------------
    # Validate expected structure
    # ---------------------------------------------------------

    if "entities" not in graph_data:
        raise ValueError(
            "Graph response does not contain 'entities'."
        )

    if "relationships" not in graph_data:
        raise ValueError(
            "Graph response does not contain 'relationships'."
        )

    if not isinstance(graph_data["entities"], list):
        raise ValueError(
            "'entities' must be a list."
        )

    if not isinstance(
        graph_data["relationships"],
        list,
    ):
        raise ValueError(
            "'relationships' must be a list."
        )

    return graph_data


def extract_graph_from_chunk(
    chunk: Document,
) -> dict[str, Any]:
    """
    Extract graph information from one LangChain Document
    chunk while preserving the original chunk metadata.
    """

    graph_data = extract_graph_from_text(
        chunk.page_content
    )

    graph_data["metadata"] = (
        chunk.metadata.copy()
    )

    return graph_data


# -------------------------------------------------------------
# Local test
# -------------------------------------------------------------

if __name__ == "__main__":

    TEST_PDF = (
        "data/uploads/"
        "spotify_web_app_architecture.pdf"
    )

    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH EXTRACTION TEST")
    print("=" * 70)

    try:

        # -----------------------------------------------------
        # Step 1 - Load PDF
        # -----------------------------------------------------

        print("\n1. Loading PDF...")

        documents = load_pdf(
            TEST_PDF
        )

        print(
            f"   Pages loaded: {len(documents)}"
        )

        # -----------------------------------------------------
        # Step 2 - Chunk PDF
        # -----------------------------------------------------

        print("\n2. Chunking PDF...")

        chunks = chunk_documents(
            documents
        )

        print(
            f"   Chunks available: {len(chunks)}"
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated."
            )

        # -----------------------------------------------------
        # Step 3 - Test only Chunk 1
        # -----------------------------------------------------

        test_chunk = chunks[0]

        print("\n3. Testing Chunk 1 only...")

        print(
            f"   Page: "
            f"{test_chunk.metadata.get('page')}"
        )

        print(
            f"   Chunk ID: "
            f"{test_chunk.metadata.get('chunk_id')}"
        )

        print(
            f"   Characters: "
            f"{len(test_chunk.page_content)}"
        )

        # -----------------------------------------------------
        # Step 4 - LLM graph extraction
        # -----------------------------------------------------

        print(
            "\n4. Calling LLM for entity and "
            "relationship extraction..."
        )

        graph = extract_graph_from_chunk(
            test_chunk
        )

        # -----------------------------------------------------
        # Display entities
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("ENTITIES")
        print("=" * 70)

        entities = graph["entities"]

        print(
            f"\nTotal entities: {len(entities)}\n"
        )

        for entity in entities:

            print(
                f"- {entity.get('name')} "
                f"({entity.get('type')})"
            )

        # -----------------------------------------------------
        # Display relationships
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("RELATIONSHIPS")
        print("=" * 70)

        relationships = graph[
            "relationships"
        ]

        print(
            f"\nTotal relationships: "
            f"{len(relationships)}\n"
        )

        for relationship in relationships:

            print(
                f"{relationship.get('source')} "
                f"--["
                f"{relationship.get('relationship')}"
                f"]--> "
                f"{relationship.get('target')}"
            )

        # -----------------------------------------------------
        # Display source metadata
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("SOURCE METADATA")
        print("=" * 70)

        print(
            graph["metadata"]
        )

        print("\n" + "=" * 70)
        print(
            "KNOWLEDGE GRAPH EXTRACTION "
            "TEST COMPLETED"
        )
        print("=" * 70)

    except Exception as error:

        print("\nKnowledge Graph extraction failed.")

        print(
            f"Error: {error}"
        )

        raise