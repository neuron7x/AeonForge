import os, socket, time, http.client

def wait_port(host, port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def test_postgres_up():
    assert wait_port("127.0.0.1", 5432), "Postgres did not become healthy on :5432"

def test_redis_up():
    assert wait_port("127.0.0.1", 6379), "Redis did not become healthy on :6379"

def test_neo4j_http_up():
    assert wait_port("127.0.0.1", 7474), "Neo4j HTTP :7474 not listening"
    conn = http.client.HTTPConnection("127.0.0.1", 7474, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.status in (200, 301, 302), f"Unexpected Neo4j HTTP status: {resp.status}"
