"""
榕能电力审图知识库 — Gradio 交互界面
HTTP 客户端模式，所有操作通过 FastAPI 完成
"""

import os
import time
import requests
import gradio as gr

API_BASE = "http://localhost:8000"


def _api(method: str, path: str, **kwargs) -> dict | list | str:
    """通用 API 请求"""
    url = f"{API_BASE}{path}"
    timeout = kwargs.pop("timeout", 120)
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code >= 400:
            detail = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return f"[ERROR] {resp.status_code}: {detail}"
        if not resp.text or not resp.text.strip():
            return f"[ERROR] API 返回空响应 (HTTP {resp.status_code})"
        return resp.json()
    except requests.ConnectionError:
        return f"[ERROR] 无法连接后端 ({API_BASE})，请先启动 API 服务"
    except requests.Timeout:
        return f"[ERROR] API 请求超时 ({timeout}s)"
    except Exception as e:
        return f"[ERROR] 请求失败: {e}"


# ========== 文件缓存 ==========
_file_cache = {"files": [], "ts": 0}

def _refresh_file_cache():
    """刷新文件列表缓存（默认排除已删除文件）"""
    result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if not isinstance(result, str):
        all_files = result.get("files", [])
        # 默认排除已删除的文件
        _file_cache["files"] = [f for f in all_files if f.get("status") != "deleted"]
        _file_cache["ts"] = time.time()
    return _file_cache["files"]


# ========== 智能问答 ==========

def rag_ask(query: str, domain: str, top_k: int):
    if not query.strip():
        return "请输入问题", ""
    t0 = time.time()

    domain_filter = None if domain == "全部" else domain
    result = _api("POST", "/ask", json={
        "query": query, "top_k": top_k, "domain_filter": domain_filter,
    })

    elapsed = (time.time() - t0) * 1000

    if isinstance(result, str):
        return result, ""

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    sources = result.get("sources", [])

    source_lines = []
    for s in sources[:10]:
        fname = os.path.basename(s.get("file_path", ""))
        source_lines.append(f"- **{s.get('doc_number') or '无编号'}** | {s.get('domain','?')}/{s.get('category','?')} | `{fname}`")

    info = f"[API] 耗时: {elapsed:.0f}ms | 引用: {len(citations)} 条 | 来源: {len(sources)} 个\n\n" + "\n".join(source_lines)
    return answer, info


# ========== 文档检索 ==========

def search_only(query: str, domain: str, top_k: int):
    if not query.strip():
        return "请输入查询内容"
    t0 = time.time()

    domain_filter = None if domain == "全部" else domain
    result = _api("POST", "/search", json={
        "query": query, "top_k": top_k, "domain_filter": domain_filter,
    })

    elapsed = (time.time() - t0) * 1000

    if isinstance(result, str):
        return result

    domain_label = {"变电": "[变]", "配电": "[配]", "送电输电": "[送]", "综合": "[综]"}
    lines = [
        f"### 检索: _{query}_",
        f"**类型:** {result.get('query_type','?')} | **候选:** {result.get('total_candidates',0)} | **耗时:** {elapsed:.0f}ms",
        f"---",
    ]
    for i, item in enumerate(result.get("results", [])):
        dm = item.get("domain", "")
        lbl = domain_label.get(dm, "[?]")
        fname = os.path.basename(item.get("file_path", ""))
        if not fname:
            fname = item.get("doc_number", "") or "未知文件"
        lines.append(f"### {lbl} [{i+1}] {fname}")
        lines.append(f"**{dm}/{item.get('category','-')}** | 电压:{item.get('voltage_level') or '-'} | 发布:{item.get('publish_level') or '-'}")
        lines.append(f"> {item.get('text','')[:300]}")
        lines.append("")

    if not result.get("results"):
        lines.append("[INFO] 知识库为空，请先上传文件。")

    return "\n".join(lines)


# ========== 文件上传入库 ==========

def upload_and_index(files, domain: str, category: str):
    if not files:
        return "[WARN] 请先选择文件"

    results = []
    ok_count = 0
    for f in files:
        try:
            orig_name = os.path.basename(f.name)
            with open(f.name, "rb") as fh:
                form_data = {}
                if domain and domain != "自动":
                    form_data["domain"] = domain
                if category:
                    form_data["category"] = category
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (orig_name, fh)},
                    data=form_data,
                    timeout=300,
                )
            if resp.status_code == 200:
                r = resp.json()
                ok_count += 1
                icon = "[OK]" if r.get("success") else "[WARN]"
                chunk_info = f"chunks:{r.get('chunks_created',0)}"
                time_info = f"{r.get('total_time_ms',0):.0f}ms"
                results.append(f"| {icon} {orig_name[:50]} | {chunk_info} | {time_info} |")
                msg = r.get("error_message", "")
                if msg:
                    results.append(f"|   {msg[:120]} |")
            else:
                results.append(f"| [ERR] {orig_name[:40]} | HTTP {resp.status_code} |")
        except Exception as e:
            results.append(f"| [ERR] {os.path.basename(f.name)[:40]}: {str(e)[:80]} |")

    if not results:
        return "[WARN] 无文件被处理"

    header = [f"## 入库结果 ({ok_count}/{len(files)} 成功)", f"| 文件 | 详情 |", f"|---|---|"]
    return "\n".join(header + results)


# ========== 文件管理 ==========

def _build_file_tree(files: list) -> str:
    """构建文件夹树状视图"""
    if not files:
        return "[INFO] 暂无入库文件，请先上传文档。"

    # 按域分组
    by_domain = {}
    for f in sorted(files, key=lambda x: (x.get("domain",""), x.get("category",""), x.get("file_name",""))):
        dm = f.get("domain", "") or "未分类"
        cat = f.get("category", "") or "未分类"
        if dm not in by_domain:
            by_domain[dm] = {}
        if cat not in by_domain[dm]:
            by_domain[dm][cat] = []
        by_domain[dm][cat].append(f)

    domain_emoji = {"变电": "B", "配电": "P", "送电输电": "S", "综合": "Z", "未分类": "?"}
    status_icon = {"completed": "O", "processing": ".", "failed": "X", "deleted": "D"}

    total_files = len(files)
    total_chunks = sum(f.get("chunks_count", 0) for f in files)

    lines = [
        f"## 文件管理",
        f"**{total_files}** 个文件 | **{total_chunks}** 个chunks",
        f"---",
    ]

    for dm in sorted(by_domain.keys()):
        cats = by_domain[dm]
        dm_count = sum(len(v) for v in cats.values())
        emoji = domain_emoji.get(dm, "?")
        lines.append(f"### [{emoji}] {dm} ({dm_count} 文件)")

        for cat in sorted(cats.keys()):
            cat_files = cats[cat]
            lines.append(f"**{cat}** ({len(cat_files)})")
            for f in cat_files:
                st = f.get("status", "?")
                icon = status_icon.get(st, "?")
                fname = f.get("file_name", "?")[:60]
                chunks = f.get("chunks_count", 0)
                created = f.get("created_at", "")[:10] if f.get("created_at") else ""
                lines.append(f"- `[{icon}]` {fname} | chunks:{chunks} | {created}")
        lines.append("")

    lines.append(f"---\n*刷新时间: {time.strftime('%H:%M:%S')}*")
    return "\n".join(lines)


def refresh_file_list(status_filter: str, domain_filter: str, search: str):
    """带过滤的刷新文件列表"""
    # 直接从 API 获取（不过滤 deleted，让状态过滤自己决定）
    result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if isinstance(result, str):
        return f"[ERROR] {result}", gr.update()
    files = result.get("files", [])

    # 默认排除已删除
    if status_filter == "全部" or not status_filter:
        files = [f for f in files if f.get("status") != "deleted"]
    elif status_filter:
        files = [f for f in files if f.get("status") == status_filter]
    if domain_filter and domain_filter != "全部":
        files = [f for f in files if f.get("domain") == domain_filter]
    if search and search.strip():
        kw = search.strip().lower()
        files = [f for f in files if kw in f.get("file_name","").lower() or kw in f.get("doc_number","").lower()]

    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name","")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    tree = _build_file_tree(files)
    return tree, gr.update(choices=choices, value=None)


def delete_selected(file_id: str):
    """删除选中文件"""
    if not file_id or not file_id.strip():
        return "[WARN] 请先选择一个文件", gr.update(), gr.update()

    result = _api("DELETE", f"/files/{file_id.strip()}")
    if isinstance(result, str):
        return result, gr.update(), gr.update()

    # 重新从 API 获取（排除已删除）
    api_result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if isinstance(api_result, str):
        files = []
    else:
        files = [f for f in api_result.get("files", []) if f.get("status") != "deleted"]

    tree = _build_file_tree(files)
    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name","")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    return f"[OK] 已删除: {file_id[:16]}...", tree, gr.update(choices=choices, value=None)


def view_file_details(file_id: str):
    """查看文件详情"""
    if not file_id or not file_id.strip():
        return "[INFO] 请在左侧选择一个文件查看详情"

    files = _file_cache.get("files", [])
    target = None
    for f in files:
        if f.get("file_hash") == file_id.strip() or f.get("file_name") == file_id.strip():
            target = f
            break

    if not target:
        return f"[WARN] 未找到: {file_id}"

    lines = [
        f"### 文件详情",
        f"| 属性 | 值 |",
        f"|---|---|",
        f"| 文件名 | {target.get('file_name','?')} |",
        f"| Hash | `{target.get('file_hash','?')[:32]}...` |",
        f"| 大小 | {target.get('file_size',0):,} bytes |",
        f"| 状态 | {target.get('status','?')} |",
        f"| Chunks | {target.get('chunks_count',0)} |",
        f"| 域 | {target.get('domain','-')} |",
        f"| 类目 | {target.get('category','-')} |",
        f"| 编号 | {target.get('doc_number','-')} |",
        f"| 路径 | `{target.get('original_path','?')}` |",
        f"| 入库时间 | {target.get('created_at','?')} |",
    ]
    return "\n".join(lines)


def refresh_stats():
    result = _api("GET", "/files/summary")
    if isinstance(result, str):
        return result, 0, 0, {}

    total = result.get("total_files", 0)
    chunks = result.get("total_chunks", 0)
    by_domain = result.get("by_domain", {})
    by_status = result.get("by_status", {})

    lines = [
        "### 入库统计",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 总文件数 | {total} |",
        f"| 已完成 | {by_status.get('completed',0)} |",
        f"| 失败 | {by_status.get('failed',0)} |",
        f"| 总 Chunks | {chunks:,} |",
        f"| 总字符数 | {result.get('total_chars',0):,} |",
        "",
        "**按域分布**",
    ]
    for dm, cnt in sorted(by_domain.items(), key=lambda x: -x[1]):
        lines.append(f"- {dm}: {cnt} 个文件")

    # 饼图数据
    return "\n".join(lines), total, chunks, by_domain


# ========== UI ==========

with gr.Blocks(title="榕能电力审图知识库 RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 榕能电力审图知识库 — 智能问答系统")

    # 初始化文件缓存
    _refresh_file_cache()

    with gr.Tabs():
        # ===== Tab 1: 智能问答 =====
        with gr.TabItem("智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    qa_query = gr.Textbox(label="输入问题", lines=3,
                        placeholder="变电消防要求？10kV配电安全距离？")
                    with gr.Row():
                        qa_domain = gr.Dropdown(choices=["全部","变电","配电","送电输电","综合"],
                            value="全部", label="专业域过滤", scale=2)
                        qa_topk = gr.Slider(5, 30, 15, 5, label="参考文档数", scale=1)
                    qa_btn = gr.Button("提问", variant="primary", size="lg")
                with gr.Column(scale=2):
                    qa_info = gr.Markdown("")
            qa_answer = gr.Markdown("> 等待提问...")
            qa_btn.click(fn=rag_ask, inputs=[qa_query, qa_domain, qa_topk],
                        outputs=[qa_answer, qa_info])

        # ===== Tab 2: 文件入库 =====
        with gr.TabItem("文件入库"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 上传文件")
                    upload_files = gr.File(label="选择文件",
                        file_count="multiple",
                        file_types=[".pdf",".doc",".docx",".xls",".xlsx",".txt",".md",".ofd",".ppt",".pptx"])
                    with gr.Row():
                        upload_domain = gr.Dropdown(choices=["自动","变电","配电","送电输电","综合"],
                            value="自动", label="域")
                        upload_category = gr.Textbox(label="类目", placeholder="如: 标准规范")
                    upload_btn = gr.Button("上传并入库", variant="primary")
                    upload_result = gr.Markdown("等待上传...")
                with gr.Column(scale=1):
                    gr.Markdown("### 入库说明")
                    gr.Markdown("""
                    - 支持 PDF/DOC/DOCX/XLS/XLSX/PPT/PPTX/TXT/MD/OFD
                    - 可多选文件批量上传
                    - 域和类目为空时自动识别
                    - 已入库文件自动跳过
                    """)

            upload_btn.click(fn=upload_and_index,
                inputs=[upload_files, upload_domain, upload_category],
                outputs=[upload_result])

        # ===== Tab 3: 文件管理 =====
        with gr.TabItem("文件管理"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        file_status_filter = gr.Dropdown(
                            choices=["全部","completed","failed","deleted"],
                            value="全部", label="状态", scale=1)
                        file_domain_filter = gr.Dropdown(
                            choices=["全部","变电","配电","送电输电","综合"],
                            value="全部", label="域", scale=1)
                        file_search = gr.Textbox(label="搜索", placeholder="文件名/编号", scale=2)
                    with gr.Row():
                        refresh_btn = gr.Button("刷新列表", variant="primary")
                    file_tree = gr.Markdown("点击刷新查看文件...")

                with gr.Column(scale=1):
                    gr.Markdown("### 文件操作")
                    file_selector = gr.Dropdown(label="选择文件", choices=[], interactive=True)
                    file_detail = gr.Markdown("[INFO] 选择文件后点击查看详情")
                    with gr.Row():
                        view_btn = gr.Button("查看详情")
                        delete_btn = gr.Button("删除文件", variant="stop")
                    op_result = gr.Markdown("")

            refresh_btn.click(fn=refresh_file_list,
                inputs=[file_status_filter, file_domain_filter, file_search],
                outputs=[file_tree, file_selector])
            view_btn.click(fn=view_file_details, inputs=[file_selector], outputs=[file_detail])
            delete_btn.click(fn=delete_selected, inputs=[file_selector],
                outputs=[op_result, file_tree, file_selector])

            # 页面加载时自动刷新
            demo.load(fn=refresh_file_list,
                inputs=[file_status_filter, file_domain_filter, file_search],
                outputs=[file_tree, file_selector])

        # ===== Tab 4: 文档检索 =====
        with gr.TabItem("文档检索"):
            with gr.Row():
                with gr.Column(scale=3):
                    srch_query = gr.Textbox(label="检索关键词", lines=2,
                        placeholder="仅在向量库中搜索，不生成回答")
                with gr.Column(scale=1):
                    srch_domain = gr.Dropdown(choices=["全部","变电","配电","送电输电","综合"],
                        value="全部", label="域过滤")
                    srch_topk = gr.Slider(3, 30, 10, 1, label="结果数")
            srch_btn = gr.Button("检索", variant="primary")
            srch_output = gr.Markdown("等待检索...")
            srch_btn.click(fn=search_only, inputs=[srch_query, srch_domain, srch_topk],
                          outputs=[srch_output])

        # ===== Tab 5: 统计 =====
        with gr.TabItem("统计"):
            stats_btn = gr.Button("刷新统计")
            stats_output = gr.Markdown("点击刷新...")
            stats_file_count = gr.Number(label="文件数", visible=False)
            stats_chunk_count = gr.Number(label="Chunks", visible=False)
            stats_btn.click(fn=refresh_stats, outputs=[stats_output, stats_file_count, stats_chunk_count])

    gr.Markdown("---\n*榕能电力审图知识库 RAG v2.0*")

if __name__ == "__main__":
    print("RAG UI (API client mode)")
    print(f"  Backend: {API_BASE}")
    print("  UI: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
