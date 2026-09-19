from neo4j import GraphDatabase

from backend.config.settings import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )

    def verify_connection(self):
        """
        Verify that the application can connect to Neo4j.
        """
        self.driver.verify_connectivity()
        print("Neo4j connection successful.")

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()


if __name__ == "__main__":
    client = Neo4jClient()

    try:
        client.verify_connection()
    finally:
        client.close()