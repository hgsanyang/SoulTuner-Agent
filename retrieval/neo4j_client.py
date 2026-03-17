import os
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    """Neo4j 图数据库连接与查询客户端"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'driver'):
            return
            
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "12345678")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: dict = None):
        """执行 Cypher 查询"""
        if not self.driver:
            logger.error("Neo4j driver is not initialized.")
            return []
            
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            return []

def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()
