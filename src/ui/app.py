"""
榕能电力审图知识库 — Gradio 交互界面
- 💬 智能问答: RAG QA
- 📤 文件入库: 单文件/批量上传 + 管理
- 📄 文档检索: 纯检索查看
- 📊 统计信息: 入库和检索统计
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from retrieval.retriever import Retriever
from ingestion.file_processor import FileProcessor, FileStatus

_retriever = None
_processor = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_processor():
    global _processor
    if _processor is None:
        _processor = FileProcessor()
    return _processor


# ========== 智能问答 ==========

def rag_ask(query: str, domain: str, top_k: int):
    if not query.strip():
        return "请输入问题", ""
    r = get_retriever()
    domain_filter = domain if domain != "全部" else None
    t0 = time.time()
    resp = r.search(query=query, top_k=top_k, domain_filter=domain_filter)

    if not resp.results:
        return "未找到相关内容", ""

    context = r.format_context_for_llm(resp.results, max_chunks=top_k)

    try:
        from generation.llm_engine import LLMEngine
        llm = LLMEngine()
        answer = llm.generate_rag_answer(query=query, context=context, query_type=resp.query_type)
        citations = llm.extract_citations(answer)
    except Exception as e:
        answer = f"⚠ LLM 不可用: {e}\n\n## 检索到 {resp.total_candidates} 条结果\n"
        for i, rr in enumerate(resp.results[:5]):
            answer += f"\n**{i+1}. {rr.doc_number or rr.file_path}**\n> {rr.text[:300]}...\n"
        citations = []

    elapsed = (time.time() - t0) * 1000
    seen = set()
    sources = []
    for item in resp.results[:10]:
        k = item.file_path
        if k not in seen:
            seen.add(k)
            sources.append(f"- **{item.doc_number or '无编号'}** | {item.domain}/{item.category} | `{item.file_path}`")

    info = (
        f"**类型:** {resp.query_type} | **域:** {resp.domain or '自动'} | "
        f"**候选:** {resp.total_candidates} | **耗时:** {elapsed:.0f}ms\n\n"
        + "\n".join(sources)
    )
    return answer, info


def search_only(query: str, domain: str, top_k: int):
    if not query.strip():
        return "请输入查询内容"
    r = get_retriever()
    domain_filter = domain if domain != "全部" else None
    t0 = time.time()
    resp = r.search(query=query, top_k=top_k, domain_filter=domain_filter)
    elapsed = (time.time() - t0) * 1000

    domain_emoji = {"变电": "🔴", "配电": "🟡", "送电输电": "🟢", "综合": "🔵"}
    lines = [
        f"### 🔍 检索: _{query}_",
        f"**类型:** {resp.query_type} | **域过滤:** {resp.domain or '无'} | "
        f"**候选数:** {resp.total_candidates} | **耗时:** {elapsed:.0f}ms",
        f"---",
    ]
    for i, item in enumerate(resp.results):
        emoji = domain_emoji.get(item.domain, "⚪")
        lines.append(
            f"### {emoji} [{i+1}] {item.doc_number or item.file_path}")
        lines.append(
            f"**{item.domain}/{item.category}** | "
            f"电压:{item.voltage_level or '-'} | "
            f"发布:{item.publish_level or '-'} | 页码:{item.page_num or '-'}")
        lines.append(f"> {item.text[:300]}{'...' if len(item.text)>300 else ''}")
        lines.append("")
    return "\n".join(lines)


# ========== 文件上传入库 ==========

def upload_and_index(files, domain: str, category: str):
    """上传单个或多个文件并入库"""
    if not files:
        return "⚠ 请先选择文件"

    processor = get_processor()
    paths = []

    for f in files:
        src = f.name  # Gradio 临时文件路径
        # 复制到 uploads 目录
        safe_name = os.path.basename(src).replace(" ", "_")
        dest = os.path.join(str(processor.uploads_dir), safe_name)
        if os.path.exists(dest):
            base, ext = os.path.splitext(safe_name)
            dest = os.path.join(str(processor.uploads_dir), f"{base}_{int(time.time())}{ext}")
        import shutil
        shutil.copy2(src, dest)
        paths.append(dest)

    batch = processor.process_batch(paths, domain=domain or None, category=category or None)

    lines = [
        f"## 📤 入库结果",
        f"| # | 文件名 | 状态 | Chunks | 域 | 耗时 |",
        f"|---|---|---|---|---|---|"
    ]
    for i, r in enumerate(batch.results):
        status_icon = "✅" if r.status == FileStatus.COMPLETED else "❌"
        lines.append(
            f"| {i+1} | {r.file_name[:40]} | {status_icon} {r.status.value} | "
            f"{r.chunks_created} | {r.domain or '-'} | {r.total_time_ms:.0f}ms |"
        )
        if r.error_message:
            lines.append(f"| | ⚠ _{r.error_message[:100]}_ |||||")

    lines.append(f"\n**合计:** {batch.total} 个文件, 成功 {batch.success}, 失败 {batch.failed}")
    return "\n".join(lines)


def list_indexed_files(status_filter: str, domain_filter: str):
    """列出已入库文件"""
    processor = get_processor()
    st = status_filter if status_filter != "全部" else None
    dm = domain_filter if domain_filter != "全部" else None
    files = processor.list_files(status=st, domain=dm, limit=50)

    if not files:
        return "暂无入库文件"

    lines = [
        f"## 📋 已入库文件 ({len(files)} 条)",
        f"| # | 文件名 | 状态 | Chunks | 域/类目 | 入库时间 |",
        f"|---|---|---|---|---|---|"
    ]
    for i, f in enumerate(files[:30]):
        status_icon = {"completed": "✅", "processing": "⏳", "failed": "❌", "deleted": "🗑️"}.get(f.get("status"), "❓")
        lines.append(
            f"| {i+1} | {f['file_name'][:35]} | {status_icon} {f['status']} | "
            f"{f['chunks_count']} | {f.get('domain','-')}/{f.get('category','-')} | {f.get('created_at','')[:16]} |"
        )
    return "\n".join(lines)


def delete_file_by_name(file_name: str):
    """按文件名删除"""
    if not file_name.strip():
        return "⚠ 请输入文件名或 hash"
    processor = get_processor()
    ok = processor.delete(file_name.strip())
    return f"✅ 已删除: {file_name}" if ok else f"❌ 未找到: {file_name}"


def reindex_file_by_name(file_name: str):
    """按文件名重建索引"""
    if not file_name.strip():
        return "⚠ 请输入文件名或 hash"
    processor = get_processor()
    result = processor.reindex(file_name.strip())
    if result.status == FileStatus.COMPLETED:
        return f"✅ 重建完成: {result.file_name}, {result.chunks_created} chunks, {result.total_time_ms:.0f}ms"
    return f"❌ 重建失败: {result.error_message}"


def refresh_summary():
    """刷新入库统计"""
    try:
        processor = get_processor()
        s = processor.get_summary()
        lines = [
            "### 📊 入库统计",
            f"| 指标 | 值 |",
            f"|---|---|",
            f"| 总文件数 | {s['total_files']} |",
            f"| 已完成 | {s['by_status'].get('completed', 0)} |",
            f"| 处理中 | {s['by_status'].get('processing', 0)} |",
            f"| 失败 | {s['by_status'].get('failed', 0)} |",
            f"| 已删除 | {s['by_status'].get('deleted', 0)} |",
            f"| 总 Chunks | {s['total_chunks']:,} |",
            f"| 总字符数 | {s['total_chars']:,} |",
            f"",
            "**按域分布**",
        ]
        for dm, cnt in sorted(s.get('by_domain', {}).items(), key=lambda x: -x[1]):
            lines.append(f"- {dm}: {cnt} 个文件")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠ {e}"


# ========== UI ==========

with gr.Blocks(title="榕能电力审图知识库 RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 榕能电力审图知识库 — 智能问答系统
    基于 **50,000+ 电力设计文件** 的模块化 RAG 知识库 · 支持单选/多选文件入库
    """)

    with gr.Tabs():
        # ===== Tab 1: 智能问答 =====
        with gr.TabItem("💬 智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    qa_query = gr.Textbox(
                        label="输入问题", lines=3,
                        placeholder="变电消防要求？10kV配电安全距离？闽电发展〔2015〕241号？")
                    with gr.Row():
                        qa_domain = gr.Dropdown(
                            choices=["全部", "变电", "配电", "送电输电", "综合"],
                            value="全部", label="专业域过滤", scale=2)
                        qa_topk = gr.Slider(5, 30, 15, 5, label="参考文档数", scale=1)
                    qa_btn = gr.Button("🔍 提问", variant="primary", size="lg")
                with gr.Column(scale=2):
                    qa_info = gr.Markdown("")

            qa_answer = gr.Markdown("> 等待提问...")
            qa_btn.click(fn=rag_ask, inputs=[qa_query, qa_domain, qa_topk],
                        outputs=[qa_answer, qa_info])

            gr.Markdown("""
            ### 💡 示例
            - 🔴 变电设计中有哪些关于消防的要求？
            - 🟡 10kV配电线路的安全距离是多少？
            - 🟢 架空输电线路对建筑物的最小距离？
            - 🔴 变电和配电在接地要求上有什么区别？
            """)

        # ===== Tab 2: 文件入库 =====
        with gr.TabItem("📤 文件入库"):
            with gr.Row():
                # 左侧: 上传区
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 上传文件入库")
                    upload_files = gr.File(
                        label="选择文件 (单选或多选)",
                        file_count="multiple",
                        file_types=[".pdf", ".doc", ".docx", ".xls", ".xlsx",
                                   ".txt", ".md", ".ofd", ".ppt", ".pptx"],
                    )
                    with gr.Row():
                        upload_domain = gr.Dropdown(
                            choices=["自动", "变电", "配电", "送电输电", "综合"],
                            value="自动", label="指定域 (可选)")
                        upload_category = gr.Textbox(
                            label="指定类目 (可选)", placeholder="如: 标准规范")
                    upload_btn = gr.Button("🚀 上传并入库", variant="primary")
                    upload_result = gr.Markdown("等待上传...")

                # 右侧: 管理区
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 文件管理")
                    with gr.Row():
                        del_name = gr.Textbox(label="文件名或 Hash", placeholder="输入要删除的文件")
                        del_btn = gr.Button("🗑️ 删除", variant="stop")
                    del_result = gr.Markdown("")

                    with gr.Row():
                        reidx_name = gr.Textbox(label="文件名或 Hash", placeholder="输入要重建索引的文件")
                        reidx_btn = gr.Button("🔄 重建索引")
                    reidx_result = gr.Markdown("")

                    gr.Markdown("---")
                    with gr.Row():
                        list_status = gr.Dropdown(
                            choices=["全部", "completed", "processing", "failed", "deleted"],
                            value="completed", label="状态过滤")
                        list_domain = gr.Dropdown(
                            choices=["全部", "变电", "配电", "送电输电", "综合"],
                            value="全部", label="域过滤")
                    refresh_list_btn = gr.Button("🔄 刷新列表")
                    file_list_output = gr.Markdown("点击刷新查看已入库文件")

            upload_btn.click(fn=upload_and_index,
                           inputs=[upload_files, upload_domain, upload_category],
                           outputs=[upload_result])
            del_btn.click(fn=delete_file_by_name, inputs=[del_name], outputs=[del_result])
            reidx_btn.click(fn=reindex_file_by_name, inputs=[reidx_name], outputs=[reidx_result])
            refresh_list_btn.click(fn=list_indexed_files,
                                 inputs=[list_status, list_domain],
                                 outputs=[file_list_output])

        # ===== Tab 3: 文档检索 =====
        with gr.TabItem("📄 文档检索"):
            with gr.Row():
                with gr.Column(scale=3):
                    srch_query = gr.Textbox(label="检索关键词", lines=2,
                                           placeholder="仅在向量库中搜索，不生成回答")
                with gr.Column(scale=1):
                    srch_domain = gr.Dropdown(
                        choices=["全部", "变电", "配电", "送电输电", "综合"],
                        value="全部", label="域过滤")
                    srch_topk = gr.Slider(5, 30, 10, 5, label="结果数")
            srch_btn = gr.Button("🔍 检索", variant="primary")
            srch_output = gr.Markdown("等待检索...")
            srch_btn.click(fn=search_only, inputs=[srch_query, srch_domain, srch_topk],
                          outputs=[srch_output])

        # ===== Tab 4: 统计 =====
        with gr.TabItem("📊 统计"):
            stats_btn = gr.Button("🔄 刷新统计")
            stats_output = gr.Markdown("点击刷新...")
            stats_btn.click(fn=refresh_summary, outputs=[stats_output])

    gr.Markdown("---\n*榕能电力审图知识库 RAG · v2.0 · BGE-M3 + Milvus + Qwen2.5-7B*")

if __name__ == "__main__":
    print("🚀 启动榕能知识库 RAG v2.0 (模块化入库版)")
    print("   UI 地址: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
