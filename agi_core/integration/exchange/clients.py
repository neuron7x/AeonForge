from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional
from urllib.parse import urlencode, urljoin

import requests


@dataclass
class OrderResponse:
    raw: Mapping[str, Any]

    @property
    def order_id(self) -> str:
        return str(self.raw.get("orderId") or self.raw.get("orderID") or self.raw.get("id"))

    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))


class ExchangeClient:
    """Base class for exchange integrations providing consistent helpers."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.session = session or requests.Session()

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[MutableMapping[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Mapping[str, Any]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = self.session.request(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    # Public API -----------------------------------------------------------

    def place_order(self, *_, **__) -> OrderResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def cancel_order(self, *_, **__) -> Mapping[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_open_orders(self, *_, **__) -> Mapping[str, Any]:  # pragma: no cover
        raise NotImplementedError


class BinanceSpotClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.binance.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(base_url, api_key, api_secret, session=session)

    def _signed_request(self, method: str, path: str, params: MutableMapping[str, Any]) -> Mapping[str, Any]:
        params.setdefault("timestamp", self._timestamp())
        query = urlencode(sorted(params.items()))
        signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.api_key}
        return self._request(method, path, params=params, headers=headers)

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        time_in_force: str | None = None,
    ) -> OrderResponse:
        params: MutableMapping[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
        if time_in_force:
            params["timeInForce"] = time_in_force
        raw = self._signed_request("POST", "/api/v3/order", params)
        return OrderResponse(raw)

    def cancel_order(self, *, symbol: str, order_id: str) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {"symbol": symbol, "orderId": order_id}
        return self._signed_request("DELETE", "/api/v3/order", params)

    def get_open_orders(self, *, symbol: str | None = None) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._signed_request("GET", "/api/v3/openOrders", params)


class BinanceFuturesClient(BinanceSpotClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://fapi.binance.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(api_key, api_secret, base_url=base_url, session=session)

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        leverage: int | None = None,
        price: float | None = None,
    ) -> OrderResponse:
        params: MutableMapping[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
            params.setdefault("timeInForce", "GTC")
        if leverage is not None:
            params["leverage"] = leverage
        raw = self._signed_request("POST", "/fapi/v1/order", params)
        return OrderResponse(raw)

    def cancel_order(self, *, symbol: str, order_id: str) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {"symbol": symbol, "orderId": order_id}
        return self._signed_request("DELETE", "/fapi/v1/order", params)

    def get_open_orders(self, *, symbol: str | None = None) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._signed_request("GET", "/fapi/v1/openOrders", params)


class BybitClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bybit.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(base_url, api_key, api_secret, session=session)

    def _signed_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        timestamp = self._timestamp()
        param_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        to_sign = timestamp + self.api_key + "10000" + param_str
        signature = hmac.new(self.api_secret, to_sign.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "10000",
            "Content-Type": "application/json",
        }
        return {"headers": headers, "data": param_str}

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResponse:
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "orderType": order_type.upper(),
            "qty": quantity,
        }
        if price is not None:
            payload["price"] = price
        signed = self._signed_payload(payload)
        raw = self._request(
            "POST",
            "/v5/order/create",
            data=signed["data"],
            headers=signed["headers"],
        )
        return OrderResponse(raw)

    def cancel_order(self, *, symbol: str, order_id: str) -> Mapping[str, Any]:
        payload = {"symbol": symbol, "orderId": order_id}
        signed = self._signed_payload(payload)
        return self._request("POST", "/v5/order/cancel", data=signed["data"], headers=signed["headers"])

    def get_open_orders(self, *, symbol: str) -> Mapping[str, Any]:
        payload = {"symbol": symbol}
        signed = self._signed_payload(payload)
        return self._request("POST", "/v5/order/realtime", data=signed["data"], headers=signed["headers"])


class OkxClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: str = "https://www.okx.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(base_url, api_key, api_secret, session=session)
        self.passphrase = passphrase

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(time.time())
        prehash = f"{timestamp}{method.upper()}{path}{body}"
        signature = base64.b64encode(hmac.new(self.api_secret, prehash.encode(), hashlib.sha256).digest()).decode()
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    def place_order(
        self,
        *,
        inst_id: str,
        side: str,
        ord_type: str,
        size: str,
        px: str | None = None,
    ) -> OrderResponse:
        body = json.dumps({"instId": inst_id, "tdMode": "cross", "side": side, "ordType": ord_type, "sz": size, "px": px})
        headers = self._headers("POST", "/api/v5/trade/order", body)
        raw = self._request("POST", "/api/v5/trade/order", data=body, headers=headers)
        return OrderResponse(raw)

    def cancel_order(self, *, inst_id: str, order_id: str) -> Mapping[str, Any]:
        body = json.dumps({"instId": inst_id, "ordId": order_id})
        headers = self._headers("POST", "/api/v5/trade/cancel-order", body)
        return self._request("POST", "/api/v5/trade/cancel-order", data=body, headers=headers)

    def get_open_orders(self, *, inst_type: str = "ANY") -> Mapping[str, Any]:
        headers = self._headers("GET", "/api/v5/trade/orders-pending")
        return self._request("GET", "/api/v5/trade/orders-pending", params={"instType": inst_type}, headers=headers)


class CoinbaseAdvancedClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: str = "https://api.coinbase.com",
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(base_url, api_key, api_secret, session=session)
        self.passphrase = passphrase

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(base64.b64decode(self.api_secret), message.encode(), hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    def place_order(
        self,
        *,
        product_id: str,
        side: str,
        order_type: str,
        size: str,
        price: str | None = None,
    ) -> OrderResponse:
        body_dict = {
            "product_id": product_id,
            "side": side,
            "type": order_type,
            "size": size,
        }
        if price is not None:
            body_dict["price"] = price
        body = json.dumps(body_dict)
        path = "/api/v3/brokerage/orders"
        headers = self._headers("POST", path, body)
        raw = self._request("POST", path, data=body, headers=headers)
        return OrderResponse(raw)

    def cancel_order(self, *, order_id: str) -> Mapping[str, Any]:
        path = f"/api/v3/brokerage/orders/{order_id}"
        headers = self._headers("DELETE", path)
        return self._request("DELETE", path, headers=headers)

    def get_open_orders(self) -> Mapping[str, Any]:
        path = "/api/v3/brokerage/orders/historical/batch"
        headers = self._headers("POST", path, "{}")
        return self._request("POST", path, data="{}", headers=headers)
