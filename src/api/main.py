"""
FastAPI 后端服务 — RAG 知识库 API

端点:
  POST /upload           单文件上传入库
  POST /upload/batch     批量上传入库
  DELETE /files/{id}     删除已入库文件
  POST /files/{id}/reindex  重建文件索引
  GET /files             列出已入库文件
  GET /files/summary     索引入库统计
  POST /search           纯检索 (不生成回答)
  POST /ask              RAG 完整问答
  GET /stats             知识库统计
"""

import sys
import os

# 注意: 已移除强制离线模式 — embedding 走魔搭 API, 但 BGE-Reranker 重排器
# 需从 HF 镜像 (hf_endpoint) 下载 ~1.1GB 模型, 离线模式会导致其加载失败。
# 模型下载到 embedding.hf_home 配置的缓存目录 (默认 D:/git/RongNengRAG/data/hf_cache)。
# 若网络下载失败, 可临时恢复: os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# 绕过 Windows 系统代理 (127.0.0.1:7890)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"

import json
import re
import time
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)
import shutil
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Body, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult, BatchResult
from ingestion.document_editor import (
    LayoutEditError,
    LayoutRevisionConflict,
    layout_revision,
)
from retrieval.retriever import Retriever, SearchResponse, RetrievalResult
from retrieval.cross_domain import retrieve_cross_domain
from generation.llm_engine import LLMEngine
from generation.conversation_manager import ConversationManager, beijing_now_display
from config import load_config
from api.security import (
    ApiKeyAuthenticator,
    AuthConfigurationError,
    AuthenticationError,
    cors_origins,
)
from api.upload_security import (
    resolve_local_import_paths,
    sanitize_upload_filename,
    upload_destination,
)
from api.download_utils import download_filename, download_media_type

# ==== 应用初始化 ====
_runtime_config = load_config()
_authenticator = ApiKeyAuthenticator.from_config(_runtime_config)
_ALLOWED_UPLOAD_EXTS = {
    ".pdf", ".doc", ".docx", ".wps", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".md", ".ofd", ".jpg", ".jpeg", ".png", ".ceb",
}
app = FastAPI(
    title="RAG 知识库 API",
    description="模块化文件入库 + 智能问答系统",
    version="2.0.0",
)
app.state.authenticator = _authenticator

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(_runtime_config),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Content-Disposition", "Content-Type"],
    allow_credentials=False,
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Fail closed for every data endpoint; health and CORS preflight stay public."""
    if request.method == "OPTIONS" or request.url.path == "/health":
        return await call_next(request)
    try:
        request.state.principal = _authenticator.authenticate(
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
        )
    except AuthConfigurationError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except AuthenticationError as exc:
        return JSONResponse(
            status_code=401,
            content={"detail": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.on_event("startup")
async def startup():
    """启动时初始化数据目录 + 预热模型"""
    from config import ensure_data_dirs, load_config, get_project_root
    cfg = load_config()
    _authenticator.ensure_configured()
    ensure_data_dirs(cfg)

    # 强制将所有缓存写到 E 盘，避免 C 盘被占满
    _hf = cfg.get("embedding", {}).get("hf_home", "E:/huggingface_cache")
    os.environ["HF_HOME"] = _hf
    os.environ.setdefault("TORCH_HOME", os.path.join(_hf, "torch"))
    os.makedirs(_hf, exist_ok=True)

    print(f"[startup] 项目根目录: {get_project_root()}")
    print(f"[startup] 数据目录: {cfg['paths']['uploads_dir']}")
    print(f"[startup] 向量库路径: {cfg['paths']['milvus_db']}")

    # 预热嵌入模型和重排序模型(避免首次请求等待)
    # 注意: OCR 不在启动时预热 — PaddleOCR 3个子模型 ~3-4GB 显存，
    #   与 BGE-M3(~2GB) + BGE-Reranker(~2GB) 同时加载会撑爆 12GB 显存
    # 注意: 不用 daemon 线程 — CUDA 在 daemon 线程中操作可能触发段错误 (torch 2.8+)
    import concurrent.futures
    def warmup():
        print("[startup] 预热嵌入模型...")
        try:
            e = get_retriever()
            _ = e.embedder.encode_query("预热测试")
            print("[startup] 嵌入+重排序模型预热完成")
        except Exception as ex:
            print(f"[startup] 模型预热跳过: {ex}")
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="warmup")
    _executor.submit(warmup)

# 全局实例 (延迟加载)
_processor: FileProcessor = None
_retriever: Retriever = None
_llm: LLMEngine = None
_conv_mgr: ConversationManager = None


def _build_file_aware_context(context: str, query: str, retriever) -> str:
    """
    当查询包含文件名时, 通过文件注册表识别并注入完整文档到 prompt。
    这是旧版 _build_focused_context 的升级版:
      - 旧版仅在 context 顶部加一句提示
      - 新版通过 FileRegistry 从向量库中拉取完整文档内容注入

    若 retriever 未就绪, 回退到纯文本匹配的聚焦提示。
    """
    if retriever is None:
        return _build_focused_context_fallback(context, query)

    try:
        ctx, match = retriever.build_context_with_file_injection(
            query=query,
            search_results=[],  # 此时不需要补充检索结果, 由调用方传入完整 context
            max_chunks=15,
        )
        # build_context_with_file_injection 返回的 context 已经包含完整文档
        # 检查是否真的有匹配
        if match is not None and ctx.strip():
            return ctx
    except Exception:
        pass

    return _build_focused_context_fallback(context, query)


def _build_focused_context_fallback(context: str, query: str) -> str:
    """
    回退方案: 纯文本模式匹配的聚焦提示。
    当文件注册表不可用或未匹配到文件时使用。
    """
    # 检测 "XX会议材料之X" 或 "XX材料之X" 模式
    m = re.search(r'(\d{2})\s*会议材料之([一二三四五六七八九十]+)', query)
    if not m:
        m = re.search(r'(\d+)\s*(会议材料|材料)', query)

    if not m:
        return context

    num = m.group(1)
    # 在 context 中找匹配的文件
    import re as re2
    target_file = None
    pattern = rf'{num}[^.]*会议材料之[^.]*\.(pdf|doc|docx)'
    match = re2.search(pattern, context)
    if match:
        target_file = match.group(0)
    else:
        # 宽泛匹配: 文件名含 num
        for line in context.split('\n'):
            if f'文件: {num}' in line and '会议材料' in line:
                target_file = line.strip()
                break

    if target_file:
        return (
            f"【重要: 用户要查询的是文件 \"{target_file}\" 的内容, "
            f"请只基于该文件的 chunks 回答, 其他文件内容仅供背景参考, 不要混淆。】\n\n"
            f"{context}"
        )

    return context


def get_processor() -> FileProcessor:
    global _processor
    if _processor is None:
        _processor = FileProcessor()
    # Keep writes and reads on the same Milvus client. Milvus Lite alias
    # visibility is not reliable across independently-created clients.
    if _retriever is not None and _retriever.store is not _processor.store:
        _retriever.store = _processor.store
    return _processor


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    # FileProcessor is the owner of index-generation publication. Reuse its
    # store whenever it already exists so reads observe the exact alias that
    # the writer just activated.
    if _processor is not None and _retriever.store is not _processor.store:
        _retriever.store = _processor.store
    return _retriever


def invalidate_retriever() -> None:
    """Drop the cached Milvus client after an index generation changes.

    FileProcessor and Retriever use separate Milvus clients. Milvus Lite can
    retain the alias target per client, so a cached Retriever may keep reading
    the previous generation after an atomic edit/reindex publish.
    """
    global _retriever
    _retriever = None


def get_llm():
    global _llm
    if _llm is None:
        try:
            _llm = LLMEngine()
        except Exception:
            _llm = None
    return _llm


def get_conv_mgr():
    global _conv_mgr
    if _conv_mgr is None:
        _conv_mgr = ConversationManager()
    return _conv_mgr


# ===== Pydantic 模型 =====

class UploadResponse(BaseModel):
    success: bool
    file_name: str
    file_hash: str
    status: str
    chunks_created: int
    chars_extracted: int
    domain: str
    category: str
    doc_number: str
    parse_time_ms: float
    embed_time_ms: float
    total_time_ms: float
    error_message: str = ""


class BatchUploadResponse(BaseModel):
    total: int
    success_count: int
    failed_count: int
    results: List[UploadResponse]


class LayoutEditOperation(BaseModel):
    page_num: int = Field(..., ge=1)
    block_index: int = Field(..., ge=0)
    op: str = Field(default="update", pattern="^(update|delete)$")
    content: str = Field(default="", max_length=65535)
    content_format: str = Field(default="text", pattern="^(text|html)$")


class DocumentEditRequest(BaseModel):
    base_revision: str = Field(..., min_length=64, max_length=64)
    edits: List[LayoutEditOperation] = Field(..., min_length=1, max_length=5000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=15, ge=1, le=50)
    domain_filter: Optional[str] = None
    return_coarse_results: bool = False


class SearchResultItem(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float
    confidence: float
    domain: str
    category: str
    file_path: str
    doc_number: str
    voltage_level: str
    publish_level: str
    page_num: int


class SearchAPIResponse(BaseModel):
    query: str
    query_type: str
    domain: Optional[str]
    total_candidates: int
    elapsed_ms: float
    filter_applied: Optional[str]
    results: List[SearchResultItem]
    coarse_results: Optional[List[SearchResultItem]] = None


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=15, ge=5, le=30)
    domain_filter: Optional[str] = None
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    sources: List[dict]
    elapsed_ms: float


class FileRegistryItem(BaseModel):
    file_hash: str
    original_path: str
    file_name: str
    file_size: int
    file_type: str
    status: str
    chunks_count: int
    domain: str
    category: str
    doc_number: str
    created_at: str
    updated_at: str


class SummaryResponse(BaseModel):
    total_files: int
    by_status: dict
    by_domain: dict
    total_chunks: int
    total_chars: int


# ===== 文件上传入库端点 =====

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    domain: Optional[str] = Form(default=None, description="手动指定专业域"),
    category: Optional[str] = Form(default=None, description="手动指定类目"),
):
    """
    上传单个文件并入库

    流程: 接收文件 → 保存到 uploads 目录 → 解析 → 分块 → 嵌入 → 入库
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    processor = get_processor()

    # 检查文件类型
    try:
        safe_name = sanitize_upload_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}。支持: {', '.join(sorted(_ALLOWED_UPLOAD_EXTS))}"
        )

    # 保存到 uploads 目录
    dest_path = str(upload_destination(processor.uploads_dir, safe_name))

    try:
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 处理入库
    result = processor.process(
        dest_path,
        domain=domain,
        category=category,
    )
    if result.status == FileStatus.COMPLETED:
        invalidate_retriever()

    return UploadResponse(
        success=result.status == FileStatus.COMPLETED,
        file_name=result.file_name,
        file_hash=result.file_hash,
        status=result.status.value,
        chunks_created=result.chunks_created,
        chars_extracted=result.chars_extracted,
        domain=result.domain,
        category=result.category,
        doc_number=result.doc_number,
        parse_time_ms=result.parse_time_ms,
        embed_time_ms=result.embed_time_ms,
        total_time_ms=result.total_time_ms,
        error_message=result.error_message,
    )


@app.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_files_batch(
    files: List[UploadFile] = File(...),
    domain: Optional[str] = Form(default=None),
    category: Optional[str] = Form(default=None),
):
    """
    批量上传文件并入库
    """
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")

    processor = get_processor()
    saved_paths = []

    for file in files:
        if not file.filename:
            continue
        try:
            safe_name = sanitize_upload_filename(file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型 {ext}")
        dest_path = str(upload_destination(processor.uploads_dir, safe_name))
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(dest_path)

    batch_result = processor.process_batch(saved_paths, domain=domain, category=category)
    if any(r.status == FileStatus.COMPLETED for r in batch_result.results):
        invalidate_retriever()

    results = []
    for r in batch_result.results:
        results.append(UploadResponse(
            success=r.status == FileStatus.COMPLETED,
            file_name=r.file_name,
            file_hash=r.file_hash,
            status=r.status.value,
            chunks_created=r.chunks_created,
            chars_extracted=r.chars_extracted,
            domain=r.domain,
            category=r.category,
            doc_number=r.doc_number,
            parse_time_ms=r.parse_time_ms,
            embed_time_ms=r.embed_time_ms,
            total_time_ms=r.total_time_ms,
            error_message=r.error_message,
        ))

    return BatchUploadResponse(
        total=batch_result.total,
        success_count=batch_result.success,
        failed_count=batch_result.failed,
        results=results,
    )


@app.post("/upload/from-paths")
async def add_files_from_paths(
    paths: List[str],
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """
    从已存在的本地文件路径批量入库 (不上传,直接指定路径)
    用于服务端已有文件的情况
    """
    processor = get_processor()
    security = _runtime_config.get("security", {})
    try:
        valid_paths = resolve_local_import_paths(
            paths,
            security.get("local_path_import_roots", []),
            bool(security.get("allow_local_path_import", False)),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    unsupported = [
        path for path in valid_paths
        if Path(path).suffix.lower() not in _ALLOWED_UPLOAD_EXTS
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                "本地导入包含不支持的文件类型: "
                f"{Path(unsupported[0]).suffix}"
            ),
        )
    batch_result = processor.process_batch(valid_paths, domain=domain, category=category)
    if any(r.status == FileStatus.COMPLETED for r in batch_result.results):
        invalidate_retriever()

    results = []
    for r in batch_result.results:
        results.append(UploadResponse(
            success=r.status == FileStatus.COMPLETED,
            file_name=r.file_name,
            file_hash=r.file_hash,
            status=r.status.value,
            chunks_created=r.chunks_created,
            chars_extracted=r.chars_extracted,
            domain=r.domain,
            category=r.category,
            doc_number=r.doc_number,
            parse_time_ms=r.parse_time_ms,
            embed_time_ms=r.embed_time_ms,
            total_time_ms=r.total_time_ms,
            error_message=r.error_message,
        ))

    return BatchUploadResponse(
        total=batch_result.total,
        success_count=batch_result.success,
        failed_count=batch_result.failed,
        results=results,
    )


# ===== 文件管理端点 =====

@app.get("/files/summary", response_model=SummaryResponse)
async def get_files_summary():
    """入库文件统计摘要 (必须在 /files/{identifier} 之前注册以避免路径冲突)"""
    processor = get_processor()
    s = processor.get_summary()
    return SummaryResponse(**s)


@app.get("/files/{identifier}/content")
async def get_file_content(
    identifier: str,
    editable: bool = Query(default=False, description="返回可原位编辑的版面 HTML"),
):
    """获取文件全部内容 (chunks + full_text)"""
    processor = get_processor()
    # 查找文件
    files = processor.list_files(limit=1000, check_existence=False, exclude_deleted=False)
    match = None
    for f in files:
        if f.get("file_hash") == identifier or f.get("file_name") == identifier:
            match = f
            break
    if not match:
        raise HTTPException(status_code=404, detail="文件未找到")

    # A persisted layout is sufficient for PDF preview.  Defer the vector
    # lookup to the legacy fallback so a stale/unavailable index cannot block
    # an already-published preview.
    full_doc = ""
    chunks_raw = []

    chunks = []
    from ingestion.vl_ocr import (
        _clean_fragmentary_html,
        _dedupe_markdown,
        extract_layout_outline,
        extract_text_outline,
        render_layout_pages_html,
    )
    from ingestion.document_editor import layout_page_texts
    layout_blocks = []
    layout_pages = []
    layout_cache = None
    outline = []
    current_layout_revision = ""
    layout_path = Path(processor.config["paths"].get("parsed_cache", "data/parsed_cache")) / f"{match.get('file_hash')}.layout.json"
    if layout_path.exists():
        try:
            layout_cache = json.loads(layout_path.read_text(encoding="utf-8"))
            current_layout_revision = layout_revision(layout_cache)
            layout_pages = render_layout_pages_html(layout_cache, editable=editable)
            outline = extract_layout_outline(layout_cache)
            if isinstance(layout_cache, dict):
                flattened = []
                for raw_page, blocks in layout_cache.items():
                    try:
                        page_num = int(raw_page) + 1
                    except (TypeError, ValueError):
                        continue
                    for block in blocks if isinstance(blocks, list) else []:
                        if isinstance(block, dict):
                            flattened.append({**block, "page": page_num})
                layout_blocks = flattened
            elif isinstance(layout_cache, list):
                layout_blocks = layout_cache
        except (OSError, ValueError):
            layout_blocks = []
            layout_pages = []
    # Use the same renderer as the live 8001 OCR comparison.  The frontend
    # should display the persisted pipeline layout, not a second approximation
    # of the same bbox/content data.
    if not layout_pages and layout_blocks:
        layout_pages = render_layout_pages_html(layout_blocks)
    layout_html = layout_pages[0]["layout_html"] if len(layout_pages) == 1 else ""
    if layout_cache is not None:
        # The persisted layout and the edited vector generation are built from
        # this exact page text.  Never append a Milvus read here: a stale
        # client/alias would otherwise make preview show both the new layout
        # and the old chunks after an edit.
        current_page_texts = layout_page_texts(layout_cache)
        chunks = [
            {
                "chunk_id": f"{match.get('file_hash')}_{page_num}",
                "text": _dedupe_markdown(_clean_fragmentary_html(text)),
                "page_num": page_num,
            }
            for page_num, text in current_page_texts
            if str(text or "").strip()
        ]
        full_doc = "\n\n".join(
            text.strip() for _, text in current_page_texts if str(text or "").strip()
        )
    else:
        r = get_retriever()
        full_doc = r.get_full_document(
            file_path=match.get("original_path") or match.get("file_name") or "",
            file_hash=match.get("file_hash") or "",
        )
        if match.get("file_hash"):
            chunks_raw = r.store.query_by_file_hash(
                match["file_hash"], sort_by_page=True
            )
        for c in chunks_raw:
            # chunks_raw 来自 Milvus query, 是 dict 而非对象, 必须用 dict 访问
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            # 预览接口再做一次兼容性清洗：历史 chunk 或旧影子代中的异常
            # OCR 片段不能因为未重新入库而继续污染页面展示。
            text = _dedupe_markdown(_clean_fragmentary_html(text))
            page_num = (c.get("page_num", 0) if isinstance(c, dict) else getattr(c, "page_num", 0)) or 0
            chunks.append({
                "chunk_id": c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", ""),
                "text": text,
                "page_num": page_num,
            })
    pages = {item["page_num"] for item in chunks}
    if not outline:
        outline = extract_text_outline(chunks, full_doc or "")

    return {
        "file_name": match.get("file_name") or "",
        "file_hash": match.get("file_hash") or "",
        "domain": match.get("domain") or "",
        "category": match.get("category") or "",
        "doc_number": match.get("doc_number") or "",
        "full_text": full_doc or "",
        "chunks": chunks,
        "layout_blocks": layout_blocks,
        "layout_html": layout_html,
        "layout_pages": layout_pages,
        "layout_revision": current_layout_revision,
        "editable": editable,
        "outline": outline,
        "total_chunks": len(chunks),
        "total_pages": max(
            [page for page in pages if page > 0]
            + [item["page_num"] for item in layout_pages],
            default=1 if layout_pages else len(pages),
        ),
    }


@app.put("/files/{identifier}/content")
async def save_file_content(identifier: str, payload: DocumentEditRequest):
    """Publish layout edits and rebuild this file's active vector chunks."""
    processor = get_processor()
    try:
        result = processor.save_layout_edits(
            identifier,
            payload.base_revision,
            [item.model_dump() for item in payload.edits],
        )
        invalidate_retriever()
        return result
    except LayoutRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LayoutEditError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/files/{identifier}/ocr-compare")
async def compare_file_ocr(identifier: str):
    """Run every source page through 8001 pipeline and 8080 bare VL OCR.

    This is a preview diagnostic endpoint: it does not write vectors or alter
    the active index. It reuses the configured OCR client protocols and returns
    page-level results for side-by-side inspection in DocumentPreview.
    """
    processor = get_processor()
    match = next(
        (f for f in processor.list_files(limit=1000, check_existence=False, exclude_deleted=False)
         if f.get("file_hash") == identifier or f.get("file_name") == identifier),
        None,
    )
    if not match:
        raise HTTPException(status_code=404, detail="文件未找到")
    source = match.get("original_path") or match.get("stored_path") or ""
    if not source or not os.path.exists(source):
        raise HTTPException(status_code=404, detail="物理文件不存在")

    from ingestion.vl_ocr import (
        VLOcrClient,
        extract_layout_outline,
        render_layout_html,
    )
    cfg = load_config().get("ocr", {})
    pipeline_cfg = cfg.get("vl", {})
    compare_cfg = cfg.get("compare", {})
    source_path = Path(source)
    is_pdf = source_path.suffix.lower() == ".pdf"
    is_ceb = source_path.suffix.lower() == ".ceb"
    pdf_page_count = 0
    image_bytes = None
    ceb_pages = []
    if is_pdf:
        import fitz
        with fitz.open(source) as doc:
            if not doc.page_count:
                raise HTTPException(status_code=422, detail="PDF 没有页面")
            pdf_page_count = doc.page_count
    elif is_ceb:
        try:
            ceb_renderer = getattr(processor, "ceb_renderer", None)
            if ceb_renderer is None:
                from ingestion.ceb_renderer import CEBRenderer
                ceb_renderer = CEBRenderer(processor.config)
            ceb_pages = ceb_renderer.render(
                source, match.get("file_hash") or processor.compute_hash(source)
            ).page_paths
            if not ceb_pages:
                raise HTTPException(status_code=422, detail="CEB 没有可渲染页面")
            pdf_page_count = len(ceb_pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"CEB 页面渲染失败: {exc}") from exc
    else:
        image_bytes = source_path.read_bytes()

    pipeline = VLOcrClient(
        base_url=pipeline_cfg.get("base_url", "http://127.0.0.1:8001"),
        model=pipeline_cfg.get("model", "paddleocr-pipeline"),
        timeout=int(pipeline_cfg.get("timeout", 180)),
        dpi=int(cfg.get("dpi", 150)),
        max_image_dim=int(cfg.get("max_image_dim", 3000)),
        max_tokens=int(pipeline_cfg.get("max_tokens", 1024)),
        protocol=pipeline_cfg.get("protocol", "pipeline"),
        endpoint=pipeline_cfg.get("endpoint", "/ocr"),
    )
    legacy = VLOcrClient(
        base_url=compare_cfg.get("legacy_base_url", "http://127.0.0.1:8080"),
        model=compare_cfg.get("legacy_model", "PaddleOCR-VL-1.6-0.9B"),
        timeout=int(pipeline_cfg.get("timeout", 180)),
        dpi=int(cfg.get("dpi", 150)),
        max_image_dim=int(cfg.get("max_image_dim", 3000)),
        max_tokens=4096,
        protocol=compare_cfg.get("legacy_protocol", "openai"),
    )

    def run(client):
        try:
            if is_pdf or is_ceb:
                if is_pdf:
                    page_texts = client.recognize_pdf_pages(
                        source,
                        list(range(pdf_page_count)),
                    )
                    layout_by_page = getattr(client, "last_layout_pages", {}) or {}
                else:
                    page_texts = {}
                    layout_by_page = {}
                    for page_idx, page_path in enumerate(ceb_pages):
                        page_texts[page_idx] = client.recognize_image(
                            Path(page_path).read_bytes()
                        ) or ""
                        layout_by_page[page_idx] = list(
                            getattr(client, "last_layout_blocks", []) or []
                        )
                page_results = []
                all_blocks = []
                text_parts = []
                for page_idx in range(pdf_page_count):
                    blocks = layout_by_page.get(page_idx)
                    if blocks is None:
                        blocks = layout_by_page.get(str(page_idx), [])
                    blocks = blocks if isinstance(blocks, list) else []
                    page_text = page_texts.get(page_idx, "") if isinstance(page_texts, dict) else ""
                    page_html = render_layout_html(blocks, page_num=page_idx + 1)
                    page_results.append({
                        "page_num": page_idx + 1,
                        "text": page_text or "",
                        "layout_blocks": blocks,
                        "layout_html": page_html,
                    })
                    all_blocks.extend(
                        {**block, "page": page_idx + 1}
                        for block in blocks
                        if isinstance(block, dict)
                    )
                    if (page_text or "").strip():
                        text_parts.append(f"--- 第 {page_idx + 1} 页 ---\n{page_text.strip()}")
                return {
                    "ok": True,
                    "text": "\n\n".join(text_parts),
                    "pages": page_results,
                    "layout_blocks": all_blocks,
                    "layout_html": page_results[0]["layout_html"] if pdf_page_count == 1 else "",
                    "outline": extract_layout_outline({
                        str(page_idx): blocks
                        for page_idx, blocks in layout_by_page.items()
                        if isinstance(blocks, list)
                    }),
                }

            text = client.recognize_image(image_bytes or b"")
            blocks = getattr(client, "last_layout_blocks", [])
            page_html = render_layout_html(blocks, page_num=1)
            return {
                "ok": True,
                "text": text,
                "layout_blocks": blocks,
                "layout_html": page_html,
                "outline": extract_layout_outline(blocks),
                "pages": [{
                    "page_num": 1,
                    "text": text or "",
                    "layout_blocks": blocks,
                    "layout_html": page_html,
                }],
            }
        except Exception as exc:
            return {"ok": False, "text": "", "pages": [], "error": str(exc)}

    pipeline_result = run(pipeline)
    return {
        "file_name": match.get("file_name") or Path(source).name,
        "file_hash": match.get("file_hash") or identifier,
        "pipeline_8001": pipeline_result,
        "bare_8080": run(legacy),
    }


@app.get("/files/subcategories")
async def get_subcategories(
    domain: str = Query(default=None),
    category: str = Query(default=None),
):
    """返回去重 (domain, category) 列表供级联选择器"""
    processor = get_processor()
    return processor.get_distinct_subcategories(domain=domain, category=category)


@app.get("/files/{identifier}")
async def get_file_detail(identifier: str):
    """获取单个文件详情 (按 hash 或文件名查找)"""
    processor = get_processor()
    files = processor.list_files(limit=1000)
    for f in files:
        if f.get("file_hash") == identifier or f.get("file_name") == identifier:
            return f
    raise HTTPException(status_code=404, detail="文件未找到")


@app.delete("/files/{identifier}")
async def delete_file(
    identifier: str,
    remove_file: bool = Query(default=True, description="是否同时删除物理文件"),
):
    """删除已入库文件 (从向量库中移除，可选清理物理文件)"""
    processor = get_processor()
    ok = processor.delete(identifier, remove_file=remove_file)
    if not ok:
        raise HTTPException(status_code=404, detail="文件未找到")
    invalidate_retriever()
    return {"status": "deleted", "identifier": identifier}


@app.patch("/files/{identifier}")
async def update_file_meta(identifier: str, payload: dict = Body(...)):
    """更新文件元数据 (domain, category, doc_number)"""
    processor = get_processor()
    result = processor.update_file_meta(identifier, payload)
    if not result["success"]:
        raise HTTPException(status_code=404, detail="文件未找到或无可更新字段")
    return result


@app.post("/files/{identifier}/rebuild-preview-assets")
async def rebuild_preview_assets(identifier: str):
    """补建图表/图片预览资源，不重新 OCR、分块或写入向量。"""
    processor = get_processor()
    files = processor.list_files(limit=1000, check_existence=False, exclude_deleted=False)
    match = next(
        (item for item in files if item.get("file_hash") == identifier or item.get("file_name") == identifier),
        None,
    )
    if not match:
        raise HTTPException(status_code=404, detail="文件未找到")
    source = match.get("original_path") or match.get("stored_path") or ""
    if not source or not os.path.exists(source):
        raise HTTPException(status_code=404, detail="物理文件不存在")
    layout_path = (
        Path(processor.config["paths"].get("parsed_cache", "data/parsed_cache"))
        / f"{match.get('file_hash')}.layout.json"
    )
    if not layout_path.exists():
        raise HTTPException(status_code=404, detail="版面缓存不存在，请先完成 OCR 解析")
    try:
        layout_cache = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"版面缓存无法读取: {exc}")

    from ingestion.vl_ocr import VLOcrClient, _attach_visual_assets

    ocr_cfg = load_config().get("ocr", {})
    vl_cfg = ocr_cfg.get("vl", {})
    client = VLOcrClient(
        base_url=vl_cfg.get("base_url", "http://localhost:8080"),
        model=vl_cfg.get("model", "paddleocr-vl-1.6"),
        timeout=vl_cfg.get("timeout", 180),
        dpi=ocr_cfg.get("dpi", 150),
        max_image_dim=ocr_cfg.get("max_image_dim", 3000),
        max_tokens=vl_cfg.get("max_tokens", 1024),
        protocol=vl_cfg.get("protocol", "openai"),
        endpoint=vl_cfg.get("endpoint", "/ocr"),
    )
    if Path(source).suffix.lower() == ".pdf" and isinstance(layout_cache, dict):
        enriched = client.enrich_layout_pages_with_visual_assets(source, layout_cache)
    elif isinstance(layout_cache, list):
        enriched = _attach_visual_assets(Path(source).read_bytes(), layout_cache)
    else:
        enriched = layout_cache
    temporary_path = layout_path.with_name(layout_path.name + ".tmp")
    temporary_path.write_text(json.dumps(enriched, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(layout_path)
    edited_layout_path = layout_path.with_name(
        f"{match.get('file_hash')}.layout.edited.json"
    )
    if edited_layout_path.exists():
        edited_temporary = edited_layout_path.with_name(edited_layout_path.name + ".tmp")
        edited_temporary.write_text(
            json.dumps(enriched, ensure_ascii=False), encoding="utf-8"
        )
        edited_temporary.replace(edited_layout_path)
    visual_count = sum(
        1
        for blocks in (enriched.values() if isinstance(enriched, dict) else [enriched])
        for block in (blocks if isinstance(blocks, list) else [])
        if isinstance(block, dict) and block.get("visual_data_uri")
    )
    return {
        "success": True,
        "file_hash": match.get("file_hash"),
        "visual_assets": visual_count,
        "message": "预览资源已补建，未重新 OCR、分块或写入向量",
    }


@app.get("/files/{identifier}/download")
async def download_file(identifier: str):
    """下载原始文件"""
    from fastapi.responses import FileResponse
    processor = get_processor()
    match = None
    # Resolve by hash directly so downloads keep working even when the
    # registry contains more than the first page of files.
    resolved_hash = processor._resolve_hash(identifier)
    if resolved_hash:
        match = processor._get_registry(resolved_hash)
    if not match:
        # Fallback for old/custom processors and filename substring requests.
        files2 = processor.list_files(limit=500, offset=0, check_existence=False, exclude_deleted=False)
        for f in files2:
            if identifier in (f.get("file_name") or ""):
                match = f
                break
    if not match:
        raise HTTPException(status_code=404, detail="文件未找到")
    path = match.get("original_path") or match.get("stored_path") or ""
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="物理文件不存在")
    fname = download_filename(match.get("file_name") or "", path)
    media_type = download_media_type(fname, path)
    return FileResponse(path, filename=fname, media_type=media_type)


@app.post("/files/sync")
async def sync_files(
    dry_run: bool = Query(default=False, description="仅扫描不清理"),
    check_milvus: bool = Query(default=False, description="同时检查 Milvus 中无对应 SQLite 记录的孤 chunk"),
):
    """
    一致性校验：扫描注册表中物理文件已消失的孤记录，
    自动从向量库和注册表清理

    - dry_run=false: 执行清理
    - dry_run=true: 仅返回孤记录列表，不执行删除
    - check_milvus=true: 同时清理 Milvus 中无 SQLite 注册的孤 chunk
    """
    processor = get_processor()
    result = processor.sync_orphans(dry_run=dry_run, check_milvus=check_milvus)
    if not dry_run:
        invalidate_retriever()
    return result


@app.post("/files/recover-pending")
async def recover_pending_files():
    """
    SSE 流式端点: 崩溃后自动恢复所有 pending 状态的文件

    事件类型:
      start       — {event, total}
      file_start  — {event, file_name, index, total}
      file_done   — {event, file_name, chunks, chars, elapsed_ms}
      file_error  — {event, file_name, error}
      done        — {event, total, success, failed}
    """
    from fastapi.responses import StreamingResponse

    def generate():
        processor = get_processor()

        # 先将残留的 processing 文件也重置为 pending (防御)
        processor._recover_stuck_files()

        pending_files = processor.list_files(status="pending", limit=1000)
        if not pending_files:
            yield f"data: {json.dumps({'event': 'done', 'total': 0, 'success': 0, 'failed': 0})}\n\n"
            return

        # 收集有效路径 (物理文件存在)
        valid = []
        for f in pending_files:
            fp = f.get("original_path") or f.get("stored_path") or ""
            if fp and os.path.exists(fp):
                valid.append((fp, f.get("file_name", os.path.basename(fp))))
            else:
                print(f"[recover] 跳过 pending 文件 (物理路径不存在): {f.get('file_name','?')}")

        if not valid:
            msg = json.dumps({"event": "done", "total": 0, "success": 0, "failed": 0,
                             "message": "待恢复文件的物理路径不存在"})
            yield f"data: {msg}\n\n"
            return

        total = len(valid)
        yield f"data: {json.dumps({'event': 'start', 'total': total})}\n\n"

        ok = 0
        fail = 0
        for i, (fp, fname) in enumerate(valid):
            yield f"data: {json.dumps({'event': 'file_start', 'file_name': fname, 'index': i, 'total': total})}\n\n"

            try:
                result = processor.process(fp)
                elapsed = result.total_time_ms
                if result.status.value == "completed":
                    ok += 1
                    invalidate_retriever()
                    done_data = json.dumps({
                        "event": "file_done", "file_name": fname,
                        "chunks": result.chunks_created, "chars": result.chars_extracted,
                        "elapsed_ms": elapsed,
                    })
                    yield f"data: {done_data}\n\n"
                else:
                    fail += 1
                    err_data = json.dumps({
                        "event": "file_error", "file_name": fname,
                        "error": result.error_message[:200],
                    })
                    yield f"data: {err_data}\n\n"
            except Exception as e:
                fail += 1
                exc_data = json.dumps({
                    "event": "file_error", "file_name": fname,
                    "error": str(e)[:200],
                })
                yield f"data: {exc_data}\n\n"

        final_data = json.dumps({
            "event": "done", "total": total,
            "success": ok, "failed": fail,
        })
        yield f"data: {final_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        },
    )


@app.post("/files/{identifier}/reindex", response_model=UploadResponse)
async def reindex_file(identifier: str):
    """重建文件索引 (删除后重新解析+嵌入+入库)"""
    processor = get_processor()
    result = processor.reindex(identifier)
    if result.status == FileStatus.COMPLETED:
        invalidate_retriever()
    print(
        f"[reindex] identifier={identifier} file={result.file_name!r} "
        f"status={result.status.value!r} success={result.status == FileStatus.COMPLETED} "
        f"chunks={result.chunks_created} error={result.error_message!r}",
        flush=True,
    )
    return UploadResponse(
        success=result.status == FileStatus.COMPLETED,
        file_name=result.file_name,
        file_hash=result.file_hash,
        status=result.status.value,
        chunks_created=result.chunks_created,
        chars_extracted=result.chars_extracted,
        domain=result.domain,
        category=result.category,
        doc_number=result.doc_number,
        parse_time_ms=result.parse_time_ms,
        embed_time_ms=result.embed_time_ms,
        total_time_ms=result.total_time_ms,
        error_message=result.error_message,
    )


@app.get("/files")
async def list_files(
    status: Optional[str] = Query(default=None, description="按状态过滤"),
    domain: Optional[str] = Query(default=None, description="按域过滤"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    include_deleted: bool = Query(default=False, description="是否包含已删除的文件"),
):
    """列出已入库文件 (默认不展示已删除文件)"""
    processor = get_processor()
    # 默认排除已删除文件；显式传 status 或 include_deleted=True 时不过滤
    exclude = (not include_deleted and status is None)
    files = processor.list_files(
        status=status, domain=domain, limit=limit, offset=offset,
        exclude_deleted=exclude,
    )
    total_count = processor.count_files(
        status=status, domain=domain, exclude_deleted=exclude,
    )
    missing_count = sum(1 for f in files if f.get("file_exists") is False)

    # 注入 Milvus 真实 chunk 数，供 UI 修正文件树统计
    try:
        store_stats = processor.store.get_collection_stats()
        milvus_chunks = store_stats.get("count", 0) if store_stats.get("exists") else None
    except Exception:
        milvus_chunks = None

    return {
        "count": total_count,
        "files": files,
        "missing_count": missing_count,
        "milvus_chunks": milvus_chunks,
    }


# ===== 检索与问答端点 =====

@app.get("/health")
async def health():
    return {"status": "ok", "service": "RAG 知识库", "version": "2.0.0"}


@app.get("/stats")
async def get_stats():
    r = get_retriever()
    s = r.store.get_collection_stats()
    return {"collection_exists": s["exists"], "chunk_count": s.get("count", 0)}


@app.get("/gpu")
async def get_gpu_status():
    """GPU 显存状态 (含 WDDM 缓存说明)"""
    try:
        from utils.gpu_monitor import get_gpu_monitor
        import torch
        m = get_gpu_monitor()
        vram = m.get_vram_info()
        # torch 实际占用 vs nvidia-smi 报告值
        torch_reserved = 0
        if torch.cuda.is_available():
            torch_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
        # nvidia-smi used 减去 torch reserved 就是 WDDM 驱动缓存 + 其他进程
        wddm_cached = max(0, vram["used_mb"] - torch_reserved - 1700)  # ~1700MB 系统桌面
        return {
            **vram,
            "usage_pct": round(vram["used_mb"] / vram["total_mb"] * 100, 1) if vram["total_mb"] else 0,
            "torch_reserved_mb": torch_reserved,
            "wddm_cached_mb": wddm_cached,
            "note": "WDDM驱动缓存不计入实际占用，新CUDA分配时自动复用" if wddm_cached > 500 else "",
        }
    except ImportError:
        return {"error": "GPU 监控未安装 (pip install nvidia-ml-py)"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(req: SearchRequest):
    r = get_retriever()
    resp = r.search(query=req.query, top_k=req.top_k, domain_filter=req.domain_filter,
                    return_coarse_results=req.return_coarse_results)
    results = []
    for i, item in enumerate(resp.results):
        results.append(SearchResultItem(
            rank=i + 1, chunk_id=item.chunk_id,
            text=item.text[:500] + ("..." if len(item.text) > 500 else ""),
            score=item.score, confidence=item.confidence,
            domain=item.domain, category=item.category,
            file_path=item.file_path, doc_number=item.doc_number,
            voltage_level=item.voltage_level, publish_level=item.publish_level,
            page_num=item.page_num,
        ))
    coarse_results = None
    if resp.coarse_results:
        coarse_results = []
        for i, item in enumerate(resp.coarse_results):
            coarse_results.append(SearchResultItem(
                rank=i + 1, chunk_id=item.chunk_id,
                text=item.text[:500] + ("..." if len(item.text) > 500 else ""),
                score=item.score, confidence=item.confidence,
                domain=item.domain, category=item.category,
                file_path=item.file_path, doc_number=item.doc_number,
                voltage_level=item.voltage_level, publish_level=item.publish_level,
                page_num=item.page_num,
            ))
    return SearchAPIResponse(
        query=resp.query, query_type=resp.query_type, domain=resp.domain,
        total_candidates=resp.total_candidates, elapsed_ms=resp.elapsed_ms,
        filter_applied=resp.filter_applied, results=results,
        coarse_results=coarse_results,
    )


def _build_sources_from_results(results, citations):
    """只保留 LLM 回答中【实际引用】的文件/chunk，而非全部检索结果"""
    if not citations or not results:
        return []
    import re
    sources = []
    seen_files = set()
    for item in results[:50]:
        cited = False
        doc = item.doc_number or ""
        fname = (item.file_path or "").replace("\\", "/").split("/")[-1]
        for cit in citations:
            if doc and doc in cit:
                cited = True
                break
            # 文件名中查找 citation 关键词 (如 "GB 50229" 匹配 "37_GB_50229-2019...pdf")
            if fname:
                for tok in re.split(r'[《》「」\s，,。、：:]+', cit):
                    if len(tok) >= 4 and tok in fname:
                        cited = True
                        break
            if cited:
                break
        if not cited:
            continue
        k = item.file_path
        fh = item.chunk_id.rsplit("_", 1)[0] if len(item.chunk_id) > 64 else ""
        if k not in seen_files:
            seen_files.add(k)
            sources.append({"file_path": k, "doc_number": item.doc_number,
                           "domain": item.domain, "category": item.category,
                           "file_hash": fh, "chunks": []})
        if len(sources[-1]["chunks"]) < 5:
            sources[-1]["chunks"].append({
                "chunk_id": item.chunk_id, "text": item.text,
                "page_num": item.page_num, "score": item.score,
            })
    return sources


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    t0 = time.time()
    r = get_retriever()

    # === 统计型查询: 全量元数据聚合, 不走语义搜索 ===
    aq = r.analyzer.analyze(req.query)
    if aq.query_type == "statistical":
        from retrieval.retriever import StatisticalResult
        stats_result = r.search_statistical(req.query, aq)
        llm = get_llm()

        if llm:
            try:
                from generation.prompt_templates import get_prompt
                answer = llm.generate_rag_answer(
                    query=req.query,
                    context=stats_result.formatted_table,
                    query_type="statistical",
                )
                citations = llm.extract_citations(answer)
            except Exception as e:
                answer = f"⚠ LLM 不可用: {e}\n\n统计结果:\n{stats_result.formatted_table}"
                citations = []
        else:
            # LLM 未部署, 直接返回统计表
            answer = f"⚠ LLM 未部署。\n\n{stats_result.formatted_table}"
            citations = []

        sources = []
        for fe in stats_result.file_list[:10]:
            sources.append({"file_path": fe["name"], "doc_number": "",
                           "domain": "", "category": ""})

        if req.conversation_id:
            conv_mgr = get_conv_mgr()
            conv_mgr.add_message(req.conversation_id, "user", req.query)
            conv_mgr.add_message(req.conversation_id, "assistant", answer, citations=citations)

        return AskResponse(
            query=req.query, answer=answer, citations=citations,
            sources=sources, elapsed_ms=(time.time() - t0) * 1000
        )

    is_cross_domain = aq.query_type == "cross_domain_comparison"
    context_domain1 = None
    context_domain2 = None
    if is_cross_domain:
        resp = retrieve_cross_domain(r, req.query, req.top_k)
        context = resp.context
        context_domain1 = resp.context_domain1
        context_domain2 = resp.context_domain2
    else:
        resp = r.search(
            query=req.query, top_k=req.top_k, domain_filter=req.domain_filter
        )

    if not resp.results and not is_cross_domain:
        # 即使搜索无结果, 也尝试文件注册表: 用户可能提到了精确的文件名
        file_match = r.detect_file_in_query(req.query)
        if file_match:
            file_path = (file_match.entry.original_path or
                         file_match.entry.stored_path or
                         file_match.entry.file_name)
            full_doc = r.get_full_document(
                file_path=file_path,
                file_hash=file_match.entry.file_hash,
            )
            if full_doc:
                context = full_doc
                # Fall through to LLM generation below
            else:
                answer = "未找到相关内容"
                if req.conversation_id:
                    conv_mgr = get_conv_mgr()
                    conv_mgr.add_message(req.conversation_id, "user", req.query)
                    conv_mgr.add_message(req.conversation_id, "assistant", answer)
                return AskResponse(query=req.query, answer=answer, citations=[], sources=[],
                                  elapsed_ms=(time.time() - t0) * 1000)
        else:
            answer = "未找到相关内容"
            if req.conversation_id:
                conv_mgr = get_conv_mgr()
                conv_mgr.add_message(req.conversation_id, "user", req.query)
                conv_mgr.add_message(req.conversation_id, "assistant", answer)
            return AskResponse(query=req.query, answer=answer, citations=[], sources=[],
                              elapsed_ms=(time.time() - t0) * 1000)
    elif not is_cross_domain:
        context, file_match = r.build_context_with_file_injection(
            query=req.query, search_results=resp.results, max_chunks=req.top_k
        )
    llm = get_llm()

    # 多轮对话: 使用完整历史消息
    if req.conversation_id and llm:
        conv_mgr = get_conv_mgr()
        conv_mgr.add_message(req.conversation_id, "user", req.query)

        try:
            # 获取历史上下文消息
            history_msgs = conv_mgr.get_context_messages(req.conversation_id)

            # 构建消息列表: 系统提示 + 历史 + 当前检索上下文
            from generation.prompt_templates import get_prompt, get_system_prompt
            messages = [{"role": "system", "content": get_system_prompt(resp.query_type)}]

            # 添加历史消息 (不包含刚添加的用户消息的最后一条)
            for hm in history_msgs[:-1]:
                messages.append(hm)

            # 最后一条用户消息附带检索上下文
            messages.append({
                "role": "user",
                "content": get_prompt(
                    resp.query_type,
                    context,
                    req.query,
                    context_domain1,
                    context_domain2,
                ),
            })

            answer = llm.generate_chat(messages, temperature=0.1)
            citations = llm.extract_citations(answer)
        except Exception as e:
            answer = f"⚠ 回答生成失败 ({type(e).__name__}): {e}\n\n检索到 {resp.total_candidates} 条"
            citations = []

        # 记录助手回复
        conv_mgr.add_message(req.conversation_id, "assistant", answer, citations=citations)

    elif llm:
        try:
            answer = llm.generate_rag_answer(
                query=req.query,
                context=context,
                query_type=resp.query_type,
                context_domain1=context_domain1,
                context_domain2=context_domain2,
            )
            citations = llm.extract_citations(answer)
        except Exception as e:
            answer = f"⚠ 回答生成失败 ({type(e).__name__}): {e}\n\n检索到 {resp.total_candidates} 条"
            citations = []
    else:
        answer = "⚠ LLM 服务不可用。请检查 LLM 服务配置。\n\n## 检索到的资料\n"
        for i, rr in enumerate(resp.results[:5]):
            answer += f"\n**{i + 1}. {rr.doc_number or rr.file_path}**\n> {rr.text[:300]}...\n"
        citations = []

    # ponytail: 只保留 LLM 回答中实际引用的文件/chunk
    sources = _build_sources_from_results(resp.results, citations)

    return AskResponse(query=req.query, answer=answer, citations=citations,
                      sources=sources, elapsed_ms=(time.time() - t0) * 1000)


# ===== 流式问答端点 =====

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE 流式 RAG 问答 (同步生成器, 避免 async 事件循环缓冲)"""
    from fastapi.responses import StreamingResponse

    def generate():
        t0 = time.time()

        # === 阶段0: 发送连接成功心跳 ===
        yield f"data: {json.dumps({'status': 'searching', 'done': False})}\n\n"

        r = get_retriever()

        # === 统计型查询: 全量元数据聚合 ===
        aq = r.analyzer.analyze(req.query)
        if aq.query_type == "statistical":
            yield f"data: {json.dumps({'status': 'thinking', 'done': False, 'sources_count': 0})}\n\n"
            from retrieval.retriever import StatisticalResult
            stats_result = r.search_statistical(req.query, aq)
            llm = get_llm()

            file_sources = [
                {"file_path": fe["name"], "doc_number": "", "domain": "", "category": ""}
                for fe in stats_result.file_list[:10]
            ]

            if llm:
                try:
                    full_answer = ""
                    for token in llm.generate_rag_answer_stream(
                        query=req.query,
                        context=stats_result.formatted_table,
                        query_type="statistical",
                    ):
                        full_answer += token
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                    citations = llm.extract_citations(full_answer)
                except Exception as e:
                    full_answer = f"⚠ 错误: {e}\n\n{stats_result.formatted_table}"
                    citations = []
            else:
                full_answer = f"⚠ LLM 未部署。\n\n{stats_result.formatted_table}"
                citations = []

            yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': file_sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"

            # ponytail: 统计查询同样记录到对话
            if req.conversation_id:
                conv_mgr = get_conv_mgr()
                conv_mgr.add_message(req.conversation_id, "user", req.query)
                conv_mgr.add_message(req.conversation_id, "assistant", full_answer, citations=citations)
            return

        is_cross_domain = aq.query_type == "cross_domain_comparison"
        context_domain1 = None
        context_domain2 = None
        if is_cross_domain:
            resp = retrieve_cross_domain(r, req.query, req.top_k)
            context = resp.context
            context_domain1 = resp.context_domain1
            context_domain2 = resp.context_domain2
        else:
            resp = r.search(
                query=req.query, top_k=req.top_k, domain_filter=req.domain_filter
            )

        if not resp.results and not is_cross_domain:
            file_match = r.detect_file_in_query(req.query)
            if file_match:
                file_path = (file_match.entry.original_path or
                             file_match.entry.stored_path or
                             file_match.entry.file_name)
                full_doc = r.get_full_document(
                    file_path=file_path,
                    file_hash=file_match.entry.file_hash,
                )
                if full_doc:
                    context = full_doc
                    # Fall through to LLM generation below
                else:
                    yield f"data: {json.dumps({'token': '未找到相关内容', 'done': True, 'citations': [], 'sources': [], 'elapsed_ms': (time.time() - t0) * 1000})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'token': '未找到相关内容', 'done': True, 'citations': [], 'sources': [], 'elapsed_ms': (time.time() - t0) * 1000})}\n\n"
                return
        elif not is_cross_domain:
            context, file_match = r.build_context_with_file_injection(
                query=req.query, search_results=resp.results, max_chunks=req.top_k
            )
        llm = get_llm()

        if not llm:
            yield f"data: {json.dumps({'token': '⚠ LLM 未部署', 'done': True})}\n\n"
            return

        # 按文件去重 sources, 附带 top5 chunk 文本供前端弹窗
        sources = []
        seen = set()
        for item in resp.results[:50]:
            k = item.file_path
            fh = item.chunk_id.rsplit("_", 1)[0] if len(item.chunk_id) > 64 else ""
            if k not in seen:
                seen.add(k)
                sources.append({"file_path": k, "doc_number": item.doc_number,
                               "domain": item.domain, "category": item.category,
                               "file_hash": fh, "chunks": []})
            if len(sources[-1]["chunks"]) < 5:
                sources[-1]["chunks"].append({
                    "chunk_id": item.chunk_id, "text": item.text,
                    "page_num": item.page_num, "score": item.score,
                })

        try:
            # === 阶段1: 检索完成, 立即发心跳通知前端"思考中" ===
            yield f"data: {json.dumps({'status': 'thinking', 'done': False, 'sources_count': len(sources)})}\n\n"

            if req.conversation_id:
                conv_mgr = get_conv_mgr()
                conv_mgr.add_message(req.conversation_id, "user", req.query)

                from generation.prompt_templates import get_prompt, get_system_prompt
                history_msgs = conv_mgr.get_context_messages(req.conversation_id)
                messages = [{"role": "system", "content": get_system_prompt(resp.query_type)}]
                for hm in history_msgs[:-1]:
                    messages.append(hm)
                messages.append({
                    "role": "user",
                    "content": get_prompt(
                        resp.query_type,
                        context,
                        req.query,
                        context_domain1,
                        context_domain2,
                    ),
                })

                full_answer = ""
                for token in llm.generate_chat_stream(messages, temperature=0.1):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                citations = llm.extract_citations(full_answer)
                conv_mgr.add_message(req.conversation_id, "assistant", full_answer, citations=citations)

                # ponytail: 用 citation 过滤 sources，只保留实际引用的 chunk
                cited_sources = _build_sources_from_results(resp.results, citations)
                yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': cited_sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"
            else:
                full_answer = ""
                for token in llm.generate_rag_answer_stream(
                    query=req.query,
                    context=context,
                    query_type=resp.query_type,
                    context_domain1=context_domain1,
                    context_domain2=context_domain2,
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                citations = llm.extract_citations(full_answer)
                cited_sources = _build_sources_from_results(resp.results, citations)
                yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': cited_sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"

        except Exception as e:
            message = f"⚠ 回答生成失败 ({type(e).__name__}): {e}"
            yield f"data: {json.dumps({'token': message, 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",  # 禁用压缩防止缓冲
        },
    )


# ===== 多轮对话端点 =====

class ConversationCreateRequest(BaseModel):
    title: str = ""


class ConversationResponse(BaseModel):
    conv_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@app.post("/conversations")
async def create_conversation(payload: dict = Body(default=None)):
    """创建新会话 (可选 JSON body: {"title": "..."})"""
    conv_mgr = get_conv_mgr()
    title = ""
    if payload and isinstance(payload, dict):
        title = payload.get("title", "")
    conv_id = conv_mgr.create_conversation(title=title)
    conv = conv_mgr.get_conversation(conv_id)
    return conv


@app.get("/conversations")
async def list_conversations():
    """列出所有会话"""
    conv_mgr = get_conv_mgr()
    return conv_mgr.list_conversations()


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """获取会话详情 (含消息历史)"""
    conv_mgr = get_conv_mgr()
    conv = conv_mgr.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除会话"""
    conv_mgr = get_conv_mgr()
    ok = conv_mgr.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "deleted", "conv_id": conv_id}


# ===== Excel 工作簿微服务代理 (新增, 不影响现有路由) =====
from .excel_proxy import excel_router as _excel_router

app.include_router(_excel_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, timeout_keep_alive=600)
