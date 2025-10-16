import os
import os
import socket
import time

import pytest
import psycopg2
import redis
from neo4j import GraphDatabase

if os.getenv("EXPECT_INFRA_SERVICES") == "1":
    pytestmark = [pytest.mark.infra]
else:
    pytestmark = [pytest.mark.infra, pytest.mark.skip("External infra stack not provisioned")] 

def wait_port(host: str, port: int, timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Port {host}:{port} not ready")

def test_postgres_connect():
    wait_port("127.0.0.1", 5432, 25.0)
    conn = psycopg2.connect("dbname=testdb user=test password=test host=127.0.0.1 port=5432")
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1
    conn.close()

def test_redis_connect():
    wait_port("127.0.0.1", 6379, 20.0)
    r = redis.Redis(host="127.0.0.1", port=6379, db=0)
    r.set("ping", "pong", ex=5)
    assert r.get("ping") == b"pong"

def test_neo4j_connect():
    wait_port("127.0.0.1", 7687, 30.0)
    uri = "bolt://127.0.0.1:7687"
    auth = ("neo4j", "test")
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        res = session.run("RETURN 1 AS x").single()
        assert res["x"] == 1
    driver.close()
