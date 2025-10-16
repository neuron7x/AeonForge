import os
import psycopg2
import redis
from neo4j import GraphDatabase

def test_postgres_connect():
    dsn = f"host={os.getenv('POSTGRES_HOST','localhost')} port={os.getenv('POSTGRES_PORT','5432')} dbname={os.getenv('POSTGRES_DB','agi')} user={os.getenv('POSTGRES_USER','agi')} password={os.getenv('POSTGRES_PASSWORD','agi')}"
    conn = psycopg2.connect(dsn)
    with conn, conn.cursor() as cur:
        cur.execute("SELECT 1;")
        assert cur.fetchone()[0] == 1
    conn.close()

def test_redis_connect():
    client = redis.Redis(host=os.getenv('REDIS_HOST','localhost'), port=int(os.getenv('REDIS_PORT','6379')), db=0)
    key = "ci:test:key"
    client.set(key, "ok", ex=10)
    assert client.get(key) == b"ok"

def test_neo4j_connect():
    uri = f"bolt://{os.getenv('NEO4J_HOST','localhost')}:{os.getenv('NEO4J_BOLT_PORT','7687')}"
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','passw0rd'))
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        result = session.run("RETURN 1 AS ok;")
        assert result.single()["ok"] == 1
    driver.close()
