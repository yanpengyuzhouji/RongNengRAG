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
import time
import shutil
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult, BatchResult
from retrieval.retriever import Retriever, SearchResponse, RetrievalResult
from generation.llm_engine import LLMEngine

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

# 全局实例 (延迟加载)
_processor: FileProcessor = None
_retriever: Retriever = None
_llm: LLMEngine = None


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
    allowed_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
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

@app.delete("/files/{identifier}")
async def delete_file(identifier: str):
    """删除已入库文件 (从向量库中移除)"""
    processor = get_processor()
    ok = processor.delete(identifier)
    if not ok:
        raise HTTPException(status_code=404, detail="文件未找到")
    return {"status": "deleted", "identifier": identifier}


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
    return {"count": len(files), "files": files}


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


@app.post("/search", response_model=SearchAPIResponse)
async def search(req: SearchRequest):
    r = get_retriever()
    resp = r.search(query=req.query, top_k=req.top_k, domain_filter=req.domain_filter)
    results = []
    for i, item in enumerate(resp.results):
        results.append(SearchResultItem(
            rank=i + 1, chunk_id=item.chunk_id,
            text=item.text[:500] + ("..." if len(item.text) > 500 else ""),
            score=item.score, domain=item.domain, category=item.category,
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
        return AskResponse(query=req.query, answer="未找到相关内容", citations=[], sources=[],
                          elapsed_ms=(time.time() - t0) * 1000)

    context = r.format_context_for_llm(resp.results, max_chunks=req.top_k)
    llm = get_llm()

    if llm:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
