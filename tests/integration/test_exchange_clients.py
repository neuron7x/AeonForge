from __future__ import annotations

import base64
from unittest import mock

import pytest

from agi_core.integration.exchange import (
    BinanceFuturesClient,
    BinanceSpotClient,
    BybitClient,
    CoinbaseAdvancedClient,
    OkxClient,
)


@pytest.fixture
def fake_response():
    response = mock.MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"orderId": "1", "status": "NEW"}
    return response


def test_binance_spot_signature(fake_response):
    session = mock.MagicMock()
    session.request.return_value = fake_response
    client = BinanceSpotClient("key", "secret", session=session)
    order = client.place_order(symbol="BTCUSDT", side="buy", order_type="limit", quantity=1.0, price=10.0)
    called_args = session.request.call_args
    assert called_args[0][0] == "POST"
    params = called_args[1]["params"]
    assert "signature" in params
    assert order.order_id == "1"


def test_binance_futures_supports_leverage(fake_response):
    session = mock.MagicMock()
    session.request.return_value = fake_response
    client = BinanceFuturesClient("key", "secret", session=session)
    client.place_order(symbol="BTCUSDT", side="SELL", order_type="LIMIT", quantity=1.0, leverage=10, price=25000.0)
    params = session.request.call_args[1]["params"]
    assert params["leverage"] == 10


def test_bybit_headers(fake_response):
    session = mock.MagicMock()
    session.request.return_value = fake_response
    client = BybitClient("key", "secret", session=session)
    client.place_order(symbol="BTCUSDT", side="Buy", order_type="Limit", quantity=1.0, price=20000.0)
    headers = session.request.call_args[1]["headers"]
    assert headers["X-BAPI-API-KEY"] == "key"
    assert "X-BAPI-SIGN" in headers


def test_okx_signature(fake_response):
    session = mock.MagicMock()
    session.request.return_value = fake_response
    client = OkxClient("key", "secret", "pass", session=session)
    client.place_order(inst_id="BTC-USDT", side="buy", ord_type="limit", size="1", px="20000")
    headers = session.request.call_args[1]["headers"]
    assert headers["OK-ACCESS-KEY"] == "key"
    assert "OK-ACCESS-SIGN" in headers


def test_coinbase_signature(fake_response):
    session = mock.MagicMock()
    session.request.return_value = fake_response
    secret = base64.b64encode(b"secret").decode()
    client = CoinbaseAdvancedClient("key", secret, "pass", session=session)
    client.place_order(product_id="BTC-USD", side="BUY", order_type="limit", size="1", price="20000")
    headers = session.request.call_args[1]["headers"]
    assert headers["CB-ACCESS-KEY"] == "key"
    assert "CB-ACCESS-SIGN" in headers
