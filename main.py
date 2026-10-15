"""
TDS GA2 - Q11 (Orders API: idempotency + cursor pagination + rate limit)
        + Q12 (Middleware stack: request-context + CORS + rate limit)
Single FastAPI service, deployable on Render.

All assigned values can be overridden with environment variables.
"""
import base64
import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------- config
TOTAL_ORDERS = int(os.getenv("TOTAL_ORDERS", "57"))            # Q11  T
ORDERS_RATE_LIMIT = int(os.getenv("ORDERS_RATE_LIMIT", "20"))  # Q11  R / 10s
PING_RATE_LIMIT = int(os.getenv("PING_RATE_LIMIT", "9"))       # Q12  B / 10s
WINDOW_SECONDS = 10

EMAIL = os.getenv("EMAIL", "your-email@example.com")           # Q12 logged-in email
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://app-d4ehzt.example.com")
EXAM_ORIGIN = os.getenv("EXAM_ORIGIN", "https://exam.sanand.workers.dev")
PING_ORIGINS = {ALLOWED_ORIGIN, EXAM_ORIGIN}

RATE_LIMITS = {"/ping": PING_RATE_LIMIT, "/orders": ORDERS_RATE_LIMIT}


def rate_group(path: str):
    for prefix in RATE_LIMITS:
        if path == prefix or path.startswith(prefix + "/"):
            return prefix
    return None


# ------------------------------------------------- Middleware 3: rate limit
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        group = rate_group(request.url.path)
        client_id = (request.headers.get("x-client-id") or "").strip()
        if request.method == "OPTIONS" or group is None or not client_id:
            return await call_next(request)

        limit = RATE_LIMITS[group]
        now = time.monotonic()
        with self.lock:
            dq = self.hits[(group, client_id)]
            while dq and now - dq[0] >= WINDOW_SECONDS:
                dq.popleft()
            if len(dq) >= limit:
                retry_after = max(1, math.ceil(WINDOW_SECONDS - (now - dq[0])))
                blocked = True
            else:
                dq.append(now)
                blocked = False
        if blocked:
            return JSONResponse(
                {
                    "detail": "Too Many Requests",
                    "request_id": getattr(request.state, "request_id", None),
                },
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


# ------------------------------------------------------ Middleware 2: CORS
class ScopedCORSMiddleware(BaseHTTPMiddleware):
    @staticmethod
    def is_allowed(origin, path):
        if not origin:
            return False
        if path == "/ping" or path.startswith("/ping/"):
            return origin in PING_ORIGINS      # strict, no wildcard
        return True                            # orders API: any browser origin

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        allowed = self.is_allowed(origin, request.url.path)

        # Preflight
        if request.method == "OPTIONS" and request.headers.get("access-control-request-method"):
            if not allowed:
                return Response(status_code=403)
            req_headers = request.headers.get(
                "access-control-request-headers",
                "Content-Type, Idempotency-Key, X-Client-Id, X-Request-ID",
            )
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": req_headers,
                    "Access-Control-Max-Age": "600",
                    "Vary": "Origin",
                },
            )

        response = await call_next(request)
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Expose-Headers"] = "X-Request-ID, Retry-After"
        if origin:
            response.headers["Vary"] = "Origin"
        return response


# ----------------------------------------- Middleware 1: request context
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = (request.headers.get("x-request-id") or "").strip() or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app = FastAPI(title="TDS GA2 Q11 + Q12")
# add_middleware: last added = outermost.
# Order of execution: context -> CORS -> rate limit -> route
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ScopedCORSMiddleware)
app.add_middleware(RequestContextMiddleware)


# ------------------------------------------------------------ Q12: /ping
@app.get("/ping")
async def ping(request: Request):
    return {"email": EMAIL, "request_id": request.state.request_id}


# ------------------------------------------------------------ Q11: orders
CATALOG = [{"id": i, "item": f"Item {i}", "status": "confirmed"} for i in range(1, TOTAL_ORDERS + 1)]

_idem_lock = threading.Lock()
_idem_store = {}                     # idempotency key -> order
_next_order_id = TOTAL_ORDERS + 1    # created orders never collide with the catalog


@app.post("/orders")
async def create_order(request: Request):
    global _next_order_id
    key = (request.headers.get("idempotency-key") or "").strip()
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        body = {}

    with _idem_lock:
        if key and key in _idem_store:
            return JSONResponse(_idem_store[key], status_code=200)
        order = {"id": _next_order_id, "status": "created", **{k: v for k, v in body.items() if k != "id"}}
        _next_order_id += 1
        if key:
            _idem_store[key] = order
    return JSONResponse(order, status_code=201)


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> int:
    pad = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode(cursor + pad).decode()
    if not raw.startswith("o:"):
        raise ValueError("bad cursor")
    offset = int(raw[2:])
    if offset < 0:
        raise ValueError("bad cursor")
    return offset


@app.get("/orders")
async def list_orders(limit: int = 10, cursor: str = None):
    limit = max(1, min(limit, 100))
    offset = 0
    if cursor:
        try:
            offset = decode_cursor(cursor)
        except Exception:
            return JSONResponse({"detail": "invalid cursor"}, status_code=400)
    items = CATALOG[offset: offset + limit]
    end = offset + len(items)
    next_cursor = encode_cursor(end) if end < len(CATALOG) else None
    return {"items": items, "next_cursor": next_cursor}


@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["POST /orders", "GET /orders", "GET /ping"]}
