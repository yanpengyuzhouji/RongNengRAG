"""
FastAPI 后端服务 — 榕能电力审图知识库 RAG API

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
import json
import re
import time
import shutil
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult, BatchResult
from retrieval.retriever import Retriever, SearchResponse, RetrievalResult
from generation.llm_engine import LLMEngine
from generation.conversation_manager import ConversationManager, beijing_now_display

# ==== 应用初始化 ====
app = FastAPI(
    title="榕能电力审图知识库 RAG API",
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
    print(f"[startup] 项目根目录: {get_project_root()}")
    print(f"[startup] 数据目录: {cfg['paths']['uploads_dir']}")
    print(f"[startup] 向量库路径: {cfg['paths']['milvus_db']}")

    # 预热嵌入模型和重排序模型(避免首次请求等待)
    # 注意: OCR 不在启动时预热 — PaddleOCR 3个子模型 ~3-4GB 显存，
    #   与 BGE-M3(~2GB) + BGE-Reranker(~2GB) 同时加载会撑爆 12GB 显存
    import threading
    def warmup():
        print("[startup] 预热嵌入模型...")
        try:
            e = get_retriever()
            _ = e.embedder.encode_query("预热测试")
            print("[startup] 嵌入+重排序模型预热完成")
        except Exception as ex:
            print(f"[startup] 模型预热跳过: {ex}")
    threading.Thread(target=warmup, daemon=True).start()

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


@app.post("/files/sync")
async def sync_files(dry_run: bool = Query(default=False, description="仅扫描不清理")):
    """
    一致性校验：扫描注册表中物理文件已消失的孤记录，
    自动从向量库和注册表清理

    - dry_run=false: 执行清理
    - dry_run=true: 仅返回孤记录列表，不执行删除
    """
    processor = get_processor()
    result = processor.sync_orphans(dry_run=dry_run)
    return result


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
):
    """列出已入库文件"""
    processor = get_processor()
    files = processor.list_files(status=status, domain=domain, limit=limit, offset=offset)
    missing_count = sum(1 for f in files if f.get("file_exists") is False)
    return {"count": len(files), "files": files, "missing_count": missing_count}


@app.get("/files/summary", response_model=SummaryResponse)
async def get_files_summary():
    """入库文件统计摘要"""
    processor = get_processor()
    s = processor.get_summary()
    return SummaryResponse(**s)


# ===== 检索与问答端点 =====

@app.get("/health")
async def health():
    return {"status": "ok", "service": "榕能电力审图知识库 RAG", "version": "2.0.0"}


@app.get("/stats")
async def get_stats():
    r = get_retriever()
    s = r.store.get_collection_stats()
    return {"collection_exists": s["exists"], "chunk_count": s.get("count", 0)}


@app.get("/gpu")
async def get_gpu_status():
    """GPU 显存状态"""
    try:
        from utils.gpu_monitor import get_gpu_monitor
        m = get_gpu_monitor()
        vram = m.get_vram_info()
        return {
            **vram,
            "usage_pct": round(vram["used_mb"] / vram["total_mb"] * 100, 1) if vram["total_mb"] else 0,
            "ollama_busy": m.is_ollama_busy(),
        }
    except ImportError:
        return {"error": "GPU 监控未安装 (pip install nvidia-ml-py)"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(req: SearchRequest):
    r = get_retriever()
    resp = r.search(query=req.query, top_k=req.top_k, domain_filter=req.domain_filter)
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
    return SearchAPIResponse(
        query=resp.query, query_type=resp.query_type, domain=resp.domain,
        total_candidates=resp.total_candidates, elapsed_ms=resp.elapsed_ms,
        filter_applied=resp.filter_applied, results=results,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    t0 = time.time()
    r = get_retriever()
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
        answer = "⚠ LLM 未部署。请启动 Ollama。\n\n## 检索到的资料\n"
        for i, rr in enumerate(resp.results[:5]):
            answer += f"\n**{i + 1}. {rr.doc_number or rr.file_path}**\n> {rr.text[:300]}...\n"
        citations = []

    sources = []
    seen = set()
    for item in resp.results[:10]:
        k = item.file_path
        if k not in seen:
            seen.add(k)
            sources.append({"file_path": k, "doc_number": item.doc_number,
                           "domain": item.domain, "category": item.category})

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

        # 构建 sources
        sources = []
        seen = set()
        for item in resp.results[:10]:
            k = item.file_path
            if k not in seen:
                seen.add(k)
                sources.append({"file_path": k, "doc_number": item.doc_number,
                               "domain": item.domain, "category": item.category})

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

                yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"
            else:
                full_answer = ""
                for token in llm.generate_rag_answer_stream(
                    query=req.query, context=context, query_type=resp.query_type
                ):
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                citations = llm.extract_citations(full_answer)
                yield f"data: {json.dumps({'token': '', 'done': True, 'citations': citations, 'sources': sources, 'elapsed_ms': (time.time() - t0) * 1000, 'full_answer': full_answer})}\n\n"

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
