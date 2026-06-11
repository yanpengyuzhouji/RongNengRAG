"""
榕能电力审图知识库 — Gradio 交互界面 v3.0
- 侧边栏会话管理 + 聊天界面
- 流式生成回答
- 北京时间显示
- 多轮对话 + 会话隔离
- 文件管理 + 检索 + 统计 (保留原有功能)
"""

import os
import json
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


# ========== 会话管理 ==========

def _create_conversation(title: str = "") -> str:
    """创建新会话，返回 conv_id"""
    result = _api("POST", "/conversations", json={"title": title})
    if isinstance(result, str):
        return ""
    return result.get("conv_id", "")


def _list_conversations() -> list:
    """获取会话列表"""
    result = _api("GET", "/conversations")
    if isinstance(result, str):
        return []
    return result


def _get_conversation(conv_id: str) -> dict:
    """获取会话详情"""
    result = _api("GET", f"/conversations/{conv_id}")
    if isinstance(result, str):
        return {}
    return result


def _delete_conversation(conv_id: str) -> bool:
    """删除会话"""
    result = _api("DELETE", f"/conversations/{conv_id}")
    return not isinstance(result, str)


# ========== 聊天功能 ==========

def refresh_conv_list():
    """刷新会话列表 (sidebar)"""
    convs = _list_conversations()
    choices = []
    for c in convs:
        preview = c.get("preview", "")[:40]
        updated = c.get("updated_at", "")[:16].replace("T", " ")
        label = f"{c.get('title','?')[:25]} | {updated}"
        choices.append((label, c.get("conv_id", "")))
    if not choices:
        choices = [("(暂无会话)", "")]
    return gr.update(choices=choices, value=None)


def new_conversation():
    """创建新会话并刷新列表"""
    conv_id = _create_conversation()
    if not conv_id:
        return None, "", [], refresh_conv_list()
    # 返回新会话ID并清空聊天
    return conv_id, conv_id, [], refresh_conv_list()


def load_conversation(conv_id: str):
    """加载指定会话的历史消息到 chatbot"""
    if not conv_id:
        return [], conv_id, ""

    conv = _get_conversation(conv_id)
    if not conv:
        return [], conv_id, ""

    messages = conv.get("messages", [])
    chatbot_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        ts = m.get("timestamp", "")[:19].replace("T", " ")
        if role == "user":
            chatbot_msgs.append({"role": "user", "content": content})
        else:
            # 在助手消息末尾加时间戳
            display = content + f"\n\n---\n*{ts} (北京时间)*"
            chatbot_msgs.append({"role": "assistant", "content": display})

    return chatbot_msgs, conv_id, ""


def delete_current_conv(conv_id: str):
    """删除当前会话"""
    if not conv_id:
        return None, "", [], refresh_conv_list()
    _delete_conversation(conv_id)
    new_id = _create_conversation()
    return new_id, new_id, [], refresh_conv_list()


def chat_stream(query: str, history: list, conv_id: str, domain_filter: str, top_k: int):
    """流式聊天 — SSE token 级实时渲染"""
    if not query or not query.strip():
        yield history, conv_id
        return

    if not conv_id:
        conv_id = _create_conversation()
        if not conv_id:
            history.append({"role": "assistant", "content": "[ERROR] 无法创建会话"})
            yield history, conv_id
            return

    # 添加用户消息 + 占位 message
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": "🔍 检索中..."})
    yield history, conv_id

    # 调用 SSE 流式 API
    try:
        domain_val = None if domain_filter == "全部" else domain_filter
        resp = requests.post(
            f"{API_BASE}/ask/stream",
            json={
                "query": query,
                "top_k": top_k,
                "domain_filter": domain_val,
                "conversation_id": conv_id,
            },
            stream=True,
            timeout=(120, 900),
        )

        full_answer = ""
        thinking_shown = False
        searching_shown = True
        last_yield = time.time()

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore")
            if line_str.startswith("data: "):
                try:
                    data = json.loads(line_str[6:])
                    token = data.get("token", "")
                    done = data.get("done", False)
                    status = data.get("status", "")

                    if done:
                        if data.get("full_answer"):
                            full_answer = data.get("full_answer", full_answer)
                        break

                    if status == "searching":
                        history[-1] = {"role": "assistant", "content": "🔍 检索中..."}
                        yield history, conv_id
                        continue

                    if status == "thinking":
                        thinking_shown = True
                        searching_shown = False
                        # 心跳动画 — 每 0.5s 更新一次
                        now = time.time()
                        if now - last_yield > 0.3:
                            dots_count = ((int(now * 2) % 4) + 1)
                            dots = "." * dots_count
                            history[-1] = {"role": "assistant", "content": f"💭 思考中{dots}"}
                            yield history, conv_id
                            last_yield = now
                        continue

                    # 有实际 token 输出
                    searching_shown = False
                    if thinking_shown:
                        thinking_shown = False
                        full_answer = ""  # 清空思考占位

                    full_answer += token

                    # 限流: 最多每 0.1s 刷新一次，避免过度渲染
                    now = time.time()
                    if now - last_yield > 0.08:
                        display = full_answer + f"\n\n---\n*⏱ {time_str} (北京时间) | ⚡ 流式生成中...*"
                        history[-1] = {"role": "assistant", "content": display}
                        yield history, conv_id
                        last_yield = now

                except json.JSONDecodeError:
                    pass

        # 最终更新
        if full_answer:
            display = full_answer + f"\n\n---\n*⏱ {time_str} (北京时间)*"
            history[-1] = {"role": "assistant", "content": display}
        else:
            history[-1] = {"role": "assistant", "content": "[WARN] 未获取到回答 — 请确认知识库已入库相关文档"}

    except requests.ConnectionError:
        history[-1] = {"role": "assistant", "content": "[ERROR] 无法连接后端 API — 请确认服务已启动"}
    except requests.Timeout:
        history[-1] = {"role": "assistant", "content": "[ERROR] 请求超时 — 请重试"}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"[ERROR] {str(e)[:200]}"}

    yield history, conv_id


# ========== 文件上传入库 ==========

def upload_and_index(files, domain: str, category: str, progress=gr.Progress()):
    if not files:
        return "[WARN] 请先选择文件"

    total = len(files)
    results = []
    ok_count = 0

    progress(0, desc=f"准备入库 {total} 个文件...")

    for i, f in enumerate(files):
        orig_name = os.path.basename(f.name)
        progress((i + 1) / total, desc=f"[{i+1}/{total}] 正在处理: {orig_name[:30]}...")

        try:
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
                    timeout=600,
                )
            if resp.status_code == 200:
                r = resp.json()
                ok_count += 1
                icon = "✅" if r.get("success") else "⚠"
                chunks = r.get('chunks_created', 0)
                chars = r.get('chars_extracted', 0)
                t = r.get('total_time_ms', 0)
                results.append(
                    f"| {icon} | {orig_name[:45]} | {chunks} | {chars:,} | {t/1000:.1f}s |"
                )
                msg = r.get("error_message", "")
                if msg:
                    results.append(f"| | {msg[:100]} | | | |")
            else:
                results.append(f"| ❌ | {orig_name[:45]} | - | - | HTTP {resp.status_code} |")
                # Try to read error detail
                try:
                    err_detail = resp.json()
                    if isinstance(err_detail, dict) and 'detail' in err_detail:
                        results.append(f"| | {str(err_detail['detail'])[:100]} | | | |")
                except Exception:
                    pass
        except requests.Timeout:
            results.append(f"| ❌ | {orig_name[:45]} | - | - | 超时 (600s) |")
        except Exception as e:
            results.append(f"| ❌ | {orig_name[:40]} | - | - | {str(e)[:60]} |")

        # 实时刷新: 每次处理完一个文件就更新显示
        header = [
            f"## 入库进度 ({i+1}/{total})",
            f"| 状态 | 文件名 | Chunks | 字符数 | 耗时 |",
            f"|---|---|---:|---:|---:|",
        ]
        progress((i + 1) / total, desc=f"[{i+1}/{total}] 完成 {orig_name[:20]}")

    if not results:
        return "[WARN] 无文件被处理"

    header = [
        f"## 入库结果 ({ok_count}/{total} 成功)",
        f"| 状态 | 文件名 | Chunks | 字符数 | 耗时 |",
        f"|---|---|---:|---:|---:|",
    ]
    progress(1.0, desc="入库完成!")
    return "\n".join(header + results)


# ========== 文件管理 ==========

def _build_file_tree(files: list) -> str:
    """构建文件夹树状视图"""
    if not files:
        return "[INFO] 暂无入库文件，请先上传文档。"

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
    status_icon = {
        "completed": "O", "processing": ".", "failed": "X", "deleted": "D",
    }
    missing_icon = "~"  # 物理文件已丢失

    total_files = len(files)
    total_chunks = sum(f.get("chunks_count", 0) for f in files)
    missing_count = sum(1 for f in files if f.get("file_exists") is False)

    lines = [
        f"## 文件管理",
        f"**{total_files}** 个文件 | **{total_chunks}** 个chunks",
    ]
    if missing_count > 0:
        lines.append(f"*[WARN] {missing_count} 个文件的物理文件已丢失 — 点击「清理失效文件」移除*")
    lines.append("---")

    for dm in sorted(by_domain.keys()):
        cats = by_domain[dm]
        dm_count = sum(len(v) for v in cats.values())
        emoji = domain_emoji.get(dm, "?")
        lines.append(f"### [{emoji}] {dm} ({dm_count} 文件)")

        for cat in sorted(cats.keys()):
            cat_files = cats[cat]
            lines.append("")  # 空行确保类目间换行
            lines.append(f"**{cat}** ({len(cat_files)})")
            for f in cat_files:
                st = f.get("status", "?")
                icon = status_icon.get(st, "?")
                if f.get("file_exists") is False:
                    icon = missing_icon
                fname = f.get("file_name", "?")[:60]
                chunks = f.get("chunks_count", 0)
                created = f.get("created_at", "")[:10] if f.get("created_at") else ""
                missing_note = " [文件丢失]" if f.get("file_exists") is False else ""
                lines.append(f"- `[{icon}]` {fname}{missing_note} | chunks:{chunks} | {created}")
        lines.append("")

    lines.append(f"---\n*刷新时间: {time.strftime('%H:%M:%S')}*")
    return "\n".join(lines)


def refresh_file_list(status_filter: str, domain_filter: str, search: str):
    result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if isinstance(result, str):
        return f"[ERROR] {result}", gr.update()
    files = result.get("files", [])

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


def view_file_details(file_id: str):
    """查看文件详情 (先查 API)"""
    if not file_id or not file_id.strip():
        return "[INFO] 请在左侧选择一个文件查看详情"

    result = _api("GET", f"/files/{file_id.strip()}")
    if isinstance(result, str):
        return f"[WARN] 未找到: {file_id}"

    target = result
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


def delete_selected(file_id: str):
    if not file_id or not file_id.strip():
        return "[WARN] 请先选择一个文件", gr.update(), gr.update()

    result = _api("DELETE", f"/files/{file_id.strip()}?remove_file=true")
    if isinstance(result, str):
        return result, gr.update(), gr.update()

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


def sync_orphan_files(do_clean: bool):
    """
    一致性校验 — 扫描并清理注册表中指向不存在物理文件的孤记录
    do_clean=False: 仅扫描 (dry run)
    do_clean=True: 扫描并清理向量库
    """
    endpoint = "/files/sync?dry_run=false" if do_clean else "/files/sync?dry_run=true"
    result = _api("POST", endpoint)

    if isinstance(result, str):
        return f"[ERROR] {result}", gr.update(), gr.update()

    dry = not do_clean
    checked = result.get("total_checked", 0)
    orphan_count = result.get("orphan_count", 0)
    cleaned = result.get("cleaned", 0)

    lines = []
    if dry:
        lines.append(f"### [扫描结果] 发现 {orphan_count} 个失效文件（共检查 {checked} 个）")
    else:
        lines.append(f"### [清理完成] 已清理 {cleaned}/{orphan_count} 个失效文件（共检查 {checked} 个）")

    if "errors" in result and result["errors"]:
        lines.append(f"\n**清理异常：**")
        for e in result["errors"]:
            lines.append(f"- `[X]` {e['file_name']}: {e['error']}")

    if "orphans" in result and result["orphans"]:
        lines.append(f"\n**失效文件详情：**")
        for o in result["orphans"]:
            fname = o.get("file_name", "?")[:50]
            domain = o.get("domain", "") or "未分类"
            cat = o.get("category", "") or "未分类"
            chunks = o.get("chunks_count", 0)
            path = o.get("original_path", "")[:60]
            lines.append(f"- `[orphan]` {fname} | {domain}/{cat} | chunks:{chunks}")
            lines.append(f"  *原路径: {path}*")

    # Scan or clean完成后刷新文件列表
    api_result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if isinstance(api_result, str):
        files = []
    else:
        files = [f for f in api_result.get("files", []) if f.get("status") != "deleted"]

    tree = _build_file_tree(files)
    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name", "")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    return "\n".join(lines), tree, gr.update(choices=choices, value=None)


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
        confidence = item.get("confidence", 0)
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        lines.append(f"### {lbl} [{i+1}] {fname}")
        lines.append(f"**{dm}/{item.get('category','-')}** | 电压:{item.get('voltage_level') or '-'} | 发布:{item.get('publish_level') or '-'} | 置信度:{confidence:.0%} {conf_bar}")
        lines.append(f"> {item.get('text','')[:300]}")
        lines.append("")

    if not result.get("results"):
        lines.append("[INFO] 知识库为空，请先上传文件。")

    return "\n".join(lines)


# ========== 统计 ==========

def refresh_stats():
    result = _api("GET", "/files/summary")
    if isinstance(result, str):
        return result

    by_domain = result.get("by_domain", {})
    by_status = result.get("by_status", {})
    total = result.get("total_files", 0)
    chunks = result.get("total_chunks", 0)

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

    return "\n".join(lines)


# ========== UI 布局 ==========

CSS = """
.sidebar { padding: 10px; }
.chatbot { min-height: 500px; }
"""

with gr.Blocks(title="榕能电力审图知识库 RAG", theme=gr.themes.Soft(), css=CSS) as demo:
    # 会话状态
    current_conv = gr.State(value="")

    with gr.Tabs():
        # ===== Tab 1: 智能问答 (聊天界面) =====
        with gr.TabItem("智能问答"):
            with gr.Row():
                # 左侧会话边栏
                with gr.Column(scale=1, elem_classes="sidebar"):
                    gr.Markdown("### 会话列表")
                    new_conv_btn = gr.Button("＋ 新建会话", variant="primary", size="sm")
                    conv_list = gr.Dropdown(
                        label="选择会话",
                        choices=[("(暂无会话)", "")],
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_conv_btn = gr.Button("刷新", size="sm")
                        delete_conv_btn = gr.Button("删除会话", variant="stop", size="sm")

                # 右侧聊天区
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话",
                        type="messages",
                        height=550,
                        show_copy_button=True,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="输入问题",
                            placeholder="变电消防要求？10kV配电安全距离？",
                            scale=4,
                            lines=2,
                        )
                    with gr.Row():
                        chat_domain = gr.Dropdown(
                            choices=["全部","变电","配电","送电输电","综合"],
                            value="全部", label="专业域过滤", scale=1
                        )
                        chat_topk = gr.Slider(5, 30, 15, 5, label="参考文档数", scale=1)
                        chat_send = gr.Button("发送", variant="primary", scale=1)

            # 会话管理事件
            new_conv_btn.click(
                fn=new_conversation,
                outputs=[current_conv, conv_list, chatbot, conv_list]
            ).then(
                fn=lambda: None, outputs=[chat_input]
            )

            refresh_conv_btn.click(
                fn=refresh_conv_list,
                outputs=[conv_list]
            )

            conv_list.select(
                fn=load_conversation,
                inputs=[conv_list],
                outputs=[chatbot, current_conv, chat_input]
            )

            delete_conv_btn.click(
                fn=delete_current_conv,
                inputs=[current_conv],
                outputs=[current_conv, conv_list, chatbot, conv_list]
            )

            # 发送消息 (流式)
            chat_send.click(
                fn=chat_stream,
                inputs=[chat_input, chatbot, current_conv, chat_domain, chat_topk],
                outputs=[chatbot, current_conv]
            ).then(
                fn=lambda: "", outputs=[chat_input]
            )

            chat_input.submit(
                fn=chat_stream,
                inputs=[chat_input, chatbot, current_conv, chat_domain, chat_topk],
                outputs=[chatbot, current_conv]
            ).then(
                fn=lambda: "", outputs=[chat_input]
            )

            # 页面加载时初始化
            demo.load(
                fn=refresh_conv_list,
                outputs=[conv_list]
            )

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
                    gr.Markdown("---")
                    gr.Markdown("### 一致性维护")
                    with gr.Row():
                        scan_orphan_btn = gr.Button("扫描失效文件", variant="secondary", size="sm")
                        clean_orphan_btn = gr.Button("清理失效文件", variant="stop", size="sm")
                    op_result = gr.Markdown("")

            refresh_btn.click(fn=refresh_file_list,
                inputs=[file_status_filter, file_domain_filter, file_search],
                outputs=[file_tree, file_selector])
            view_btn.click(fn=view_file_details, inputs=[file_selector], outputs=[file_detail])
            delete_btn.click(fn=delete_selected, inputs=[file_selector],
                outputs=[op_result, file_tree, file_selector])
            scan_orphan_btn.click(fn=lambda: sync_orphan_files(False),
                outputs=[op_result, file_tree, file_selector])
            clean_orphan_btn.click(fn=lambda: sync_orphan_files(True),
                outputs=[op_result, file_tree, file_selector])

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
            stats_btn.click(fn=refresh_stats, outputs=[stats_output])

    gr.Markdown("---\n*榕能电力审图知识库 RAG v3.0 — 北京时间: {}*".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


if __name__ == "__main__":
    print("RAG UI v3.0 (streaming + multi-turn conversation)")
    print(f"  Backend: {API_BASE}")
    print("  UI: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
