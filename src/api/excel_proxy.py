"""Excel 工作簿微服务反向代理.

浏览器统一走 RongNengRAG 后端(8000)的 /excel/* 路径,此处 1:1 转发到
Excel Workbook Service(默认 http://127.0.0.1:8090,前缀 /api/v1)。

- 普通请求(上传/草稿/校验/确认/查询/报告): 整体转发,透传状态码与 content-type。
- SSE 端点(路径以 /ask 结尾): httpx 流式直通,防止缓冲。
- 不改动任何现有路由;仅新增 /excel/* 前缀。
"""

import os
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .stream_limits import StreamLimitExceeded, limited_stream

EXCEL_SERVICE_BASE = os.environ.get("EXCEL_SERVICE_BASE", "http://127.0.0.1:8090")
EXCEL_PROXY_MAX_BODY_BYTES = int(
    os.environ.get("EXCEL_PROXY_MAX_BODY_BYTES", str(210 * 1024 * 1024))
)

excel_router = APIRouter(prefix="/excel")

# 透传这些请求头(不含 host/connection 等 hop-by-hop 头)
_FORWARD_HEADERS = {
    "content-type",
    "accept",
    "accept-encoding",
    "user-agent",
    "cookie",
    "authorization",
    "x-api-key",
    "x-user-id",
    "x-request-id",
}


@excel_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def excel_proxy(path: str, request: Request):
    target = f"{EXCEL_SERVICE_BASE}/api/v1/{unquote(path)}"
    is_ask = path.rstrip("/").endswith("/ask")

    headers = {
        k: v for k, v in request.headers.items() if k.lower() in _FORWARD_HEADERS
    }
    params = dict(request.query_params)
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > EXCEL_PROXY_MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds the Excel proxy limit"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length header"}
            )

    client = httpx.AsyncClient(timeout=None if is_ask else 300)
    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                target,
                headers=headers,
                content=limited_stream(
                    request.stream(), EXCEL_PROXY_MAX_BODY_BYTES
                ),
                params=params,
            ),
            stream=True,
        )
    except StreamLimitExceeded as exc:
        await client.aclose()
        return JSONResponse(status_code=413, content={"detail": str(exc)})
    except BaseException:
        await client.aclose()
        raise

    async def gen():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    content_type = upstream.headers.get("content-type", "application/octet-stream")
    if content_type.lower().startswith("text/html") and "charset=" not in content_type.lower():
        content_type = f"{content_type}; charset=utf-8"
    response_headers = {
        "Content-Type": content_type,
        "X-Content-Type-Options": "nosniff",
    }
    if is_ask:
        response_headers.update({
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
    return StreamingResponse(
        gen(), status_code=upstream.status_code, headers=response_headers
    )
