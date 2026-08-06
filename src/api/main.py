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

# 强制离线模式 — BGE-M3 已缓存至 E:/huggingface_cache，禁止联网验证
# (Windows 系统代理会拦截所有 HTTP 请求，代理客户端未启动时导致加载失败)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 绕过 Windows 系统代理 (127.0.0.1:7890)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"

import json
import re
import time
import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)
import shutil
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult, BatchResult
from retrieval.retriever import Retriever, SearchResponse, RetrievalResult
from generation.llm_engine import LLMEngine
from generation.conversation_manager import ConversationManager, beijing_now_display

# ==== 应用初始化 ====
app = FastAPI(
    title="RAG 知识库 API",
    description="模块化文件入库 + 智能问答系统",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """启动时初始化数据目录 + 预热模型"""
    from config import ensure_data_dirs, load_config, get_project_root
    cfg = load_config()
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
    return _processor


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


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
    allowed_exts = {".pdf", ".doc", ".docx", ".wps", ".xls", ".xlsx", ".ppt", ".pptx",
                    ".txt", ".md", ".ofd", ".jpg", ".jpeg", ".png", ".ceb"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}。支持: {', '.join(sorted(allowed_exts))}"
        )

    # 保存到 uploads 目录
    safe_name = file.filename.replace(" ", "_")
    dest_path = os.path.join(str(processor.uploads_dir), safe_name)
    # 避免覆盖
    if os.path.exists(dest_path):
        base, ext_part = os.path.splitext(safe_name)
        dest_path = os.path.join(str(processor.uploads_dir),
                                 f"{base}_{int(time.time())}{ext_part}")

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
        safe_name = file.filename.replace(" ", "_")
        dest_path = os.path.join(str(processor.uploads_dir), safe_name)
        if os.path.exists(dest_path):
            base, ext_part = os.path.splitext(safe_name)
            dest_path = os.path.join(str(processor.uploads_dir),
                                     f"{base}_{int(time.time())}{ext_part}")
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(dest_path)

    batch_result = processor.process_batch(saved_paths, domain=domain, category=category)

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
    valid_paths = [p for p in paths if os.path.exists(p)]
    if not valid_paths:
        raise HTTPException(status_code=400, detail="所有路径均不存在")

    batch_result = processor.process_batch(valid_paths, domain=domain, category=category)

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
async def get_file_content(identifier: str):
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

    r = get_retriever()
    full_doc = r.get_full_document(
        file_path=match.get("original_path") or match.get("file_name") or "",
        file_hash=match.get("file_hash") or "",
    )

    # 拉 chunks 列表
    chunks_raw = []
    if match.get("file_hash"):
        chunks_raw = r.store.query_by_file_hash(match["file_hash"], sort_by_page=True)

    chunks = []
    pages = set()
    for c in chunks_raw:
        pages.add(getattr(c, "page_num", 0) or 0)
        chunks.append({
            "chunk_id": getattr(c, "chunk_id", ""),
            "text": getattr(c, "text", ""),
            "page_num": getattr(c, "page_num", 0) or 0,
        })

    return {
        "file_name": match.get("file_name") or "",
        "file_hash": match.get("file_hash") or "",
        "domain": match.get("domain") or "",
        "category": match.get("category") or "",
        "doc_number": match.get("doc_number") or "",
        "full_text": full_doc or "",
        "chunks": chunks,
        "total_chunks": len(chunks),
        "total_pages": len(pages),
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
    return {"status": "deleted", "identifier": identifier}


@app.patch("/files/{identifier}")
async def update_file_meta(identifier: str, payload: dict = Body(...)):
    """更新文件元数据 (domain, category, doc_number)"""
    processor = get_processor()
    result = processor.update_file_meta(identifier, payload)
    if not result["success"]:
        raise HTTPException(status_code=404, detail="文件未找到或无可更新字段")
    return result


@app.get("/files/{identifier}/download")
async def download_file(identifier: str):
    """下载原始文件"""
    from fastapi.responses import FileResponse
    processor = get_processor()
    files = processor.list_files(limit=1, offset=0, check_existence=False, exclude_deleted=False)
    match = None
    for f in files:
        if f.get("file_hash") == identifier:
            match = f
            break
    if not match:
        # fallback: search by filename
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
    fname = match.get("file_name") or "download"
    return FileResponse(path, filename=fname, media_type="application/octet-stream")


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

    resp = r.search(query=req.query, top_k=req.top_k, domain_filter=req.domain_filter)

    if not resp.results:
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
    else:
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
            from generation.prompt_templates import get_system_prompt
            messages = [{"role": "system", "content": get_system_prompt(resp.query_type)}]

            # 添加历史消息 (不包含刚添加的用户消息的最后一条)
            for hm in history_msgs[:-1]:
                messages.append(hm)

            # 最后一条用户消息附带检索上下文
            messages.append({
                "role": "user",
                "content": f"{context}\n\n用户问题: {req.query}\n\n请根据以上文件内容回答:"
            })

            answer = llm.generate_chat(messages, temperature=0.1)
            citations = llm.extract_citations(answer)
        except Exception as e:
            answer = f"⚠ LLM 不可用: {e}\n\n检索到 {resp.total_candidates} 条"
            citations = []

        # 记录助手回复
        conv_mgr.add_message(req.conversation_id, "assistant", answer, citations=citations)

    elif llm:
        try:
            answer = llm.generate_rag_answer(query=req.query, context=context,
                                            query_type=resp.query_type)
            citations = llm.extract_citations(answer)
        except Exception as e:
            answer = f"⚠ LLM 不可用: {e}\n\n检索到 {resp.total_candidates} 条"
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

        resp = r.search(query=req.query, top_k=req.top_k, domain_filter=req.domain_filter)

        if not resp.results:
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
        else:
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

                from generation.prompt_templates import get_system_prompt
                history_msgs = conv_mgr.get_context_messages(req.conversation_id)
                messages = [{"role": "system", "content": get_system_prompt(resp.query_type)}]
                for hm in history_msgs[:-1]:
                    messages.append(hm)
                messages.append({
                    "role": "user",
                    "content": f"{context}\n\n用户问题: {req.query}\n\n请根据以上文件内容回答:"
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
                    query=req.query, context=context, query_type=resp.query_type
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                citations = llm.extract_citations(full_answer)
                cited_sources = _build_sources_from_results(resp.results, citations)
                yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': cited_sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'⚠ 错误: {e}', 'done': True})}\n\n"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, timeout_keep_alive=600)
