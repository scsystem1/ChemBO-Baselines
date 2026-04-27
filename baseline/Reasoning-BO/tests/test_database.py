from pymilvus import connections
from neo4j import GraphDatabase
import pytest
from src.config import Config

config = Config()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_milvus_connection():
    # Host should be the IP address of the machine where the docker container is located, configured in the .env file.
    connections.connect(
        alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT
    )

    connection_list = connections.list_connections()
    assert len(connection_list) == 1
    assert connection_list[0][0] == "default"
    assert connection_list[0][1] is not None


def test_neo4j_connection():
    URI = config.NEO4J_URL
    AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
