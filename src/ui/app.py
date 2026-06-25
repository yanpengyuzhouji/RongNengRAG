"""
榕能电力审图知识库 — Gradio 交互界面 v4.0
- 自定义工业工程主题 (工程蓝 + 冷灰)
- 品牌头部 (实时统计) + 品牌页脚
- 4 Tab: 智能问答 / 文件入库 / 文件管理 / 文档检索
- SSE 流式生成 + 多轮对话 + 会话隔离
- 结构化 HTML 文件树 + 卡片式详情
- 统一状态横幅 (error/warning/success/info)
"""

import os
import json
import time
import requests
import gradio as gr

from theme import create_theme
from components import (
    render_header,
    render_footer,
    render_file_tree,
    render_file_detail,
    render_search_results,
    render_upload_progress,
    render_alert,
    render_empty_state,
)

API_BASE = "http://localhost:8000"


def _api(method: str, path: str, **kwargs) -> dict | list | str:
    """通用 API 请求"""
    url = f"{API_BASE}{path}"
    timeout = kwargs.pop("timeout", 120)
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code >= 400:
            detail = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return f"[API Error] {resp.status_code}: {detail}"
        if not resp.text or not resp.text.strip():
            return f"[API Error] 返回空响应 (HTTP {resp.status_code})"
        return resp.json()
    except requests.ConnectionError:
        return "[API Error] 无法连接后端，请先启动 API 服务 (端口 8000)"
    except requests.Timeout:
        return f"[API Error] 请求超时 ({timeout}s)"
    except Exception as e:
        return f"[API Error] 请求失败: {e}"


# ==================== 会话管理 ====================

def _create_conversation(title: str = "") -> str:
    result = _api("POST", "/conversations", json={"title": title})
    if isinstance(result, str):
        return ""
    return result.get("conv_id", "")


def _list_conversations() -> list:
    result = _api("GET", "/conversations")
    if isinstance(result, str):
        return []
    return result


def _get_conversation(conv_id: str) -> dict:
    result = _api("GET", f"/conversations/{conv_id}")
    if isinstance(result, str):
        return {}
    return result


def _delete_conversation(conv_id: str) -> bool:
    result = _api("DELETE", f"/conversations/{conv_id}")
    return not isinstance(result, str)


# ==================== 头部统计 ====================

def refresh_header():
    """刷新品牌头部统计"""
    result = _api("GET", "/files/summary")
    online = not isinstance(result, str)
    total_files = 0
    total_chunks = 0
    if online and isinstance(result, dict):
        total_files = result.get("total_files", 0)
        total_chunks = result.get("total_chunks", 0)
    return render_header(
        total_files=total_files,
        total_chunks=total_chunks,
        online=online,
    )


# ==================== 会话列表 ====================

def refresh_conv_list():
    """刷新会话列表 (sidebar)"""
    convs = _list_conversations()
    choices = []
    for c in convs:
        preview = c.get("preview", "")[:40]
        updated = c.get("updated_at", "")[:16].replace("T", " ")
        label = f"{c.get('title','?')[:25]}  |  {updated}"
        choices.append((label, c.get("conv_id", "")))
    if not choices:
        choices = [("（暂无会话）", "")]
    return gr.update(choices=choices, value=None)


def new_conversation():
    """创建新会话并刷新列表"""
    conv_id = _create_conversation()
    if not conv_id:
        return None, "", [], refresh_conv_list()
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
            display = content + f"\n\n*⏱ {ts}*"
            chatbot_msgs.append({"role": "assistant", "content": display})

    return chatbot_msgs, conv_id, ""


def delete_current_conv(conv_id: str):
    """删除当前会话"""
    if not conv_id:
        return None, "", [], refresh_conv_list()
    _delete_conversation(conv_id)
    new_id = _create_conversation()
    return new_id, new_id, [], refresh_conv_list()


# ==================== 聊天 (SSE 流式) ====================

def chat_stream(query: str, history: list, conv_id: str, domain_filter: str, top_k: int):
    """流式聊天 — SSE token 级实时渲染"""
    if not query or not query.strip():
        yield history, conv_id
        return

    if not conv_id:
        conv_id = _create_conversation()
        if not conv_id:
            history.append({"role": "assistant", "content": "⚠ 无法创建会话，请确认后端服务已启动"})
            yield history, conv_id
            return

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": "⏳ 检索中..."})
    yield history, conv_id

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
                        history[-1] = {"role": "assistant", "content": "⏳ 检索中..."}
                        yield history, conv_id
                        continue

                    if status == "thinking":
                        thinking_shown = True
                        searching_shown = False
                        now = time.time()
                        if now - last_yield > 0.3:
                            dots_count = ((int(now * 2) % 4) + 1)
                            dots = "." * dots_count
                            history[-1] = {"role": "assistant", "content": f"💭 分析中{dots}"}
                            yield history, conv_id
                            last_yield = now
                        continue

                    searching_shown = False
                    if thinking_shown:
                        thinking_shown = False
                        full_answer = ""

                    full_answer += token

                    now = time.time()
                    if now - last_yield > 0.08:
                        display = full_answer + f"\n\n*⏱ {time_str}*"
                        history[-1] = {"role": "assistant", "content": display}
                        yield history, conv_id
                        last_yield = now

                except json.JSONDecodeError:
                    pass

        # 最终更新
        if full_answer:
            display = full_answer + f"\n\n*⏱ {time_str}*"
            history[-1] = {"role": "assistant", "content": display}
        else:
            history[-1] = {"role": "assistant",
                           "content": "⚠ 未获取到回答 — 请确认知识库已入库相关文档"}

    except requests.ConnectionError:
        history[-1] = {"role": "assistant",
                       "content": "⚠ 无法连接后端 API — 请确认服务已启动 (端口 8000)"}
    except requests.Timeout:
        history[-1] = {"role": "assistant", "content": "⚠ 请求超时 — 请重试或缩小检索范围"}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"⚠ 发生错误: {str(e)[:200]}"}

    yield history, conv_id


# ==================== 文件上传入库 ====================

def upload_and_index(files, domain: str, category: str, progress=gr.Progress()):
    """文件上传并入库 — HTML 进度展示"""
    if not files:
        yield render_alert("请先选择文件", "warning")
        return

    total = len(files)
    results = []
    ok_count = 0

    progress(0, desc="准备入库...")

    for i, f in enumerate(files):
        orig_name = os.path.basename(f.name)
        progress((i + 1) / total, desc=f"[{i+1}/{total}] {orig_name[:30]}...")

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
                    timeout=(30, 900),
                )
            if resp.status_code == 200:
                r = resp.json()
                ok_count += 1
                icon = "✅" if r.get("success") else "⚠️"
                chunks = r.get("chunks_created", 0)
                chars = f"{r.get('chars_extracted', 0):,}"
                t = f"{r.get('total_time_ms', 0)/1000:.1f}s"
                results.append((icon, orig_name, chunks, chars, t))
                msg = r.get("error_message", "")
                if msg:
                    results.append(("", msg[:100], "", "", ""))
            else:
                results.append(("❌", orig_name, "-", "-", f"HTTP {resp.status_code}"))
                try:
                    err_detail = resp.json()
                    if isinstance(err_detail, dict) and "detail" in err_detail:
                        results.append(("", str(err_detail["detail"])[:100], "", "", ""))
                except Exception:
                    pass
        except requests.Timeout:
            results.append(("❌", orig_name, "-", "-", "超时"))
        except Exception as e:
            results.append(("❌", orig_name, "-", "-", str(e)[:60]))

        # 每处理完一个文件就 yield 实时更新
        yield render_upload_progress(i + 1, total, results, ok_count)

    # 全部完成
    progress(1.0, desc="入库完成!")
    yield render_upload_progress(total, total, results, ok_count, done=True)


# ==================== 崩溃恢复 ====================

def auto_recover_pending():
    """启动时自动检测并恢复 pending 文件 — HTML 进度"""
    result = _api("GET", "/files", params={"status": "pending", "limit": 100})
    if isinstance(result, str):
        yield ""
        return

    files = result.get("files", [])
    if not files:
        yield ""
        return

    total = len(files)
    fnames = [f.get("file_name", "?")[:60] for f in files]
    yield render_alert(
        f"检测到 {total} 个文件在上次异常退出时未完成入库，正在自动恢复...",
        "warning",
    )

    results = [{"name": n, "status": "⏳ 等待..."} for n in fnames]
    try:
        resp = requests.post(
            f"{API_BASE}/files/recover-pending",
            stream=True,
            timeout=(30, 1800),
        )

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore")
            if line_str.startswith("data: "):
                try:
                    data = json.loads(line_str[6:])
                    event = data.get("event", "")

                    if event == "file_start":
                        idx = data.get("index", 0)
                        if idx < len(results):
                            results[idx]["status"] = "🔄 处理中..."

                    elif event == "file_done":
                        fname = data.get("file_name", "")[:60]
                        chunks = data.get("chunks", 0)
                        chars = data.get("chars", 0)
                        elapsed = data.get("elapsed_ms", 0) / 1000
                        for r in results:
                            if r["name"] == fname:
                                r["status"] = f"✅ {chunks} chunks, {chars:,} 字符 ({elapsed:.1f}s)"
                                break

                    elif event == "file_error":
                        fname = data.get("file_name", "")[:60]
                        err = data.get("error", "")[:100]
                        for r in results:
                            if r["name"] == fname:
                                r["status"] = f"❌ {err}"
                                break

                    elif event == "done":
                        ok_count = data.get("success", 0)
                        fail_count = data.get("failed", 0)
                        if fail_count > 0:
                            yield render_alert(
                                f"恢复完成: {ok_count}/{total} 成功, {fail_count} 失败",
                                "warning",
                            )
                        else:
                            yield render_alert(
                                f"恢复完成: {total} 个文件全部成功入库",
                                "success",
                            )
                        return
                except json.JSONDecodeError:
                    pass

            processed = sum(1 for r in results if r["status"] not in ("⏳ 等待...", "🔄 处理中..."))
            in_progress = sum(1 for r in results if r["status"] == "🔄 处理中...")
            pct = int((processed + in_progress) / max(total, 1) * 100)
            bar_html = f"""<div style="margin:8px 0;">
                <div style="font-size:12px;color:#374151;margin-bottom:4px;">
                    恢复进度 ({processed + in_progress}/{total})
                </div>
                <div style="height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:#1a56db;border-radius:3px;
                        transition:width 0.3s ease;"></div>
                </div>
            </div>"""
            yield bar_html

    except requests.ConnectionError:
        yield render_alert("无法连接后端，文件恢复跳过 — 请手动重新入库", "error")
    except requests.Timeout:
        yield render_alert(f"文件恢复超时 ({total} 个文件)，请手动重新入库", "warning")
    except Exception as e:
        yield render_alert(f"文件恢复异常: {str(e)[:200]}", "error")


# ==================== 文件管理 ====================

def refresh_file_list(status_filter: str, domain_filter: str, search: str):
    """刷新文件列表 — 返回 HTML 树 + Dropdown 选项"""
    result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    if isinstance(result, str):
        return render_alert(result, "error"), gr.update()

    files = result.get("files", [])

    # 过滤
    if status_filter and status_filter != "全部":
        files = [f for f in files if f.get("status") == status_filter]
    else:
        files = [f for f in files if f.get("status") != "deleted"]
    if domain_filter and domain_filter != "全部":
        files = [f for f in files if f.get("domain") == domain_filter]
    if search and search.strip():
        kw = search.strip().lower()
        files = [f for f in files
                 if kw in f.get("file_name", "").lower()
                 or kw in f.get("doc_number", "").lower()]

    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name", "")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    tree_html = render_file_tree(files)
    return tree_html, gr.update(choices=choices, value=None)


def view_file_details(file_id: str):
    """查看文件详情 — HTML 卡片"""
    if not file_id or not file_id.strip():
        return render_empty_state("请在左侧选择一个文件查看详情")

    result = _api("GET", f"/files/{file_id.strip()}")
    if isinstance(result, str):
        return render_alert(f"未找到文件: {file_id}", "warning")

    return render_file_detail(result)


def delete_selected(file_id: str):
    """删除选中文件"""
    if not file_id or not file_id.strip():
        return render_alert("请先选择一个文件", "warning"), gr.update(), gr.update()

    result = _api("DELETE", f"/files/{file_id.strip()}?remove_file=true")
    if isinstance(result, str):
        return render_alert(result, "error"), gr.update(), gr.update()

    # 刷新列表
    api_result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    files = []
    if not isinstance(api_result, str):
        files = [f for f in api_result.get("files", []) if f.get("status") != "deleted"]

    tree_html = render_file_tree(files)
    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name", "")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    return render_alert(f"已删除文件", "success"), tree_html, gr.update(choices=choices, value=None)


def sync_orphan_files(do_clean: bool):
    """一致性校验 — 扫描/清理孤立记录"""
    endpoint = "/files/sync?dry_run=false" if do_clean else "/files/sync?dry_run=true"
    result = _api("POST", endpoint)

    if isinstance(result, str):
        return render_alert(result, "error"), gr.update(), gr.update()

    dry = not do_clean
    checked = result.get("total_checked", 0)
    orphan_count = result.get("orphan_count", 0)
    cleaned = result.get("cleaned", 0)

    if dry:
        if orphan_count == 0:
            msg = f"扫描完成 — 共检查 {checked} 个文件，未发现失效文件"
            return render_alert(msg, "success"), gr.update(), gr.update()
        else:
            msg = f"扫描完成 — 发现 {orphan_count} 个失效文件（共检查 {checked} 个），点击「清理失效文件」移除"
            return render_alert(msg, "warning"), gr.update(), gr.update()
    else:
        msg = f"清理完成 — 已清理 {cleaned}/{orphan_count} 个失效文件（共检查 {checked} 个）"
        level = "success" if cleaned > 0 else "info"
        alert_html = render_alert(msg, level)

    # 刷新列表
    api_result = _api("GET", "/files", params={"limit": 500, "offset": 0})
    files = []
    if not isinstance(api_result, str):
        files = [f for f in api_result.get("files", []) if f.get("status") != "deleted"]

    tree_html = render_file_tree(files)
    choices = []
    for f in sorted(files, key=lambda x: x.get("file_name", "")):
        label = f"{f.get('file_name','?')[:50]} [{f.get('chunks_count',0)}c]"
        choices.append((label, f.get("file_hash", "")))

    return alert_html, tree_html, gr.update(choices=choices, value=None)


# ==================== 文档检索 ====================

def search_only(query: str, domain: str, top_k: int):
    """纯向量检索 — HTML 结果卡片"""
    if not query.strip():
        return render_empty_state("请输入查询内容")

    t0 = time.time()
    domain_filter = None if domain == "全部" else domain
    result = _api("POST", "/search", json={
        "query": query, "top_k": top_k, "domain_filter": domain_filter,
    })
    elapsed = (time.time() - t0) * 1000

    if isinstance(result, str):
        return render_alert(result, "error")

    return render_search_results(query, result, elapsed)


# ==================== Tab 切换 ====================

def _on_tab_select_file_mgmt():
    """切换到文件管理 tab 时自动刷新"""
    return refresh_file_list("全部", "全部", "")


def _on_tab_select_any(*args, **kwargs):
    """切换到任意 tab 时刷新头部统计"""
    return refresh_header()


# ==================== CSS ====================

CSS = """
/* ===== 基础变量 ===== */
:root {
    --blue-600: #1a56db;
    --blue-700: #1e40af;
    --blue-50: #eff6ff;
    --slate-50: #f8fafc;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-700: #334155;
    --slate-900: #111827;
    --green-600: #059669;
    --amber-600: #d97706;
    --red-600: #dc2626;
}

/* ===== 全局: 满宽 + 可滚动 ===== */
html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    overflow-y: auto !important;  /* 根元素可滚动 */
}
body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 100vh !important; /* 至少占满视口，内容多时自然撑开 */
}
gradio-app {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    display: block !important;
}
.gradio-container {
    max-width: 100% !important;  /* 移除 Gradio 默认的固定宽度 */
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* 清理所有外层 wrapper */
.contain, .app, .main, .wrap,
.gradio-container > * {
    max-width: 100% !important;
}

body {
    font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei",
                 "PingFang SC", "Noto Sans SC", sans-serif !important;
}

/* ===== 品牌头部 (负 margin 抵消 block_padding 实现贴边) ===== */
.brand-header {
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
}
.header-stat {
    transition: transform 0.15s ease;
}

/* ===== Tab 导航: 贴在头部下方 ===== */
.tabs > .tab-nav {
    border-bottom: 2px solid var(--slate-200) !important;
    gap: 0 !important;
    padding: 0 20px !important;
    background: var(--slate-50) !important;
}
.tabs > .tab-nav button {
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--slate-700) !important;
    transition: color 0.15s, border-color 0.15s !important;
    background: transparent !important;
}
.tabs > .tab-nav button:hover {
    color: var(--blue-600) !important;
    border-bottom-color: var(--slate-200) !important;
}
.tabs > .tab-nav button.selected {
    color: var(--blue-600) !important;
    border-bottom-color: var(--blue-600) !important;
    font-weight: 600 !important;
}

/* ===== 按钮增强 ===== */
button.primary {
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
button.stop {
    font-weight: 500 !important;
}

/* ===== 输入框增强 ===== */
input:focus, textarea:focus, select:focus {
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}

/* ===== 键盘焦点环 ===== */
*:focus-visible {
    outline: 2px solid var(--blue-600) !important;
    outline-offset: 2px !important;
}

/* ===== Chatbot 消息气泡 ===== */
.chatbot .message.user {
    background: var(--blue-50) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px 8px 2px 8px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
}
.chatbot .message.bot {
    background: #fff !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: 8px 8px 8px 2px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
}

/* ===== 会话侧边栏: 左侧 Panel 风格，顶部无圆角贴边 ===== */
.sidebar {
    background: var(--slate-50) !important;
    border-radius: 0 !important;
    padding: 12px !important;
    border: none !important;
    border-right: 1px solid var(--slate-200) !important;
}

/* ===== 文件树 ===== */
.file-tree .tree-domain-header:hover {
    filter: brightness(0.96);
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* ===== 动画 ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.file-tree, .brand-header {
    animation: fadeIn 0.3s ease;
}

/* ===== 暗色模式适配 ===== */
@media (prefers-color-scheme: dark) {
    .chatbot .message.user {
        background: #1e3a5f !important;
        border-color: #1e40af !important;
        color: #dbeafe !important;
    }
    .chatbot .message.bot {
        background: #1a202c !important;
        border-color: #2d3643 !important;
        color: #e5e7eb !important;
    }
    .sidebar {
        background: #1a202c !important;
        border-color: #2d3643 !important;
    }
    .tabs > .tab-nav {
        background: #1a202c !important;
        border-bottom-color: #2d3643 !important;
    }
}

/* ===== 响应式 ===== */
@media (max-width: 768px) {
    .tabs > .tab-nav button {
        padding: 8px 12px !important;
        font-size: 13px !important;
    }
    .brand-header {
        flex-direction: column !important;
        gap: 8px !important;
    }
}
"""

# ==================== UI 布局 ====================

with gr.Blocks(
    title="榕能电力审图知识库 RAG",
    theme=create_theme(),
    css=CSS,
) as demo:

    # 会话状态
    current_conv = gr.State(value="")
    upload_status = gr.State(value="")

    # ==== 品牌头部 ====
    header_html = gr.HTML(
        value=render_header(total_files=0, total_chunks=0, online=True),
        elem_id="brand-header",
    )

    # ==== 崩溃恢复横幅 ====
    recovery_banner = gr.HTML("", visible=True, elem_id="recovery-banner")

    # ==== Tab 导航 (4 Tab) ====
    with gr.Tabs() as tabs:
        # ========== Tab 1: 智能问答 ==========
        with gr.TabItem("💬 智能问答"):
            with gr.Row():
                # 左侧会话边栏
                with gr.Column(scale=1, elem_classes="sidebar"):
                    gr.Markdown("#### 会话")
                    new_conv_btn = gr.Button("＋ 新建会话", variant="primary", size="sm")
                    conv_list = gr.Dropdown(
                        label="选择会话",
                        choices=[("（暂无会话）", "")],
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_conv_btn = gr.Button("刷新", size="sm")
                        delete_conv_btn = gr.Button("删除", variant="stop", size="sm")
                    gr.Markdown(
                        '<div style="font-size:11px;color:#9ca3af;margin-top:12px;'
                        'padding-top:8px;border-top:1px solid #e5e7eb;">'
                        '域过滤可限定检索范围<br>Top-K 越大参考越多</div>'
                    )

                # 右侧聊天区
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话",
                        type="messages",
                        height=550,
                        show_copy_button=True,
                        placeholder="输入电力设计相关问题，如：变电消防要求？10kV 配电安全距离？",
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="输入问题",
                            placeholder="变电消防要求？10kV 配电安全距离？",
                            scale=4,
                            lines=2,
                        )
                    with gr.Row():
                        chat_domain = gr.Dropdown(
                            choices=["全部", "变电", "配电", "送电输电", "综合"],
                            value="全部", label="域过滤", scale=1,
                        )
                        chat_topk = gr.Slider(5, 30, 15, 5, label="Top-K", scale=1)
                        chat_send = gr.Button("发送", variant="primary", scale=1)

            # --- 会话管理事件 ---
            new_conv_btn.click(
                fn=new_conversation,
                outputs=[current_conv, conv_list, chatbot, conv_list]
            ).then(fn=lambda: None, outputs=[chat_input])

            refresh_conv_btn.click(fn=refresh_conv_list, outputs=[conv_list])

            conv_list.select(
                fn=load_conversation,
                inputs=[conv_list],
                outputs=[chatbot, current_conv, chat_input],
            )

            delete_conv_btn.click(
                fn=delete_current_conv,
                inputs=[current_conv],
                outputs=[current_conv, conv_list, chatbot, conv_list],
            )

            # --- 发送消息 (流式) ---
            send_inputs = [chat_input, chatbot, current_conv, chat_domain, chat_topk]
            send_outputs = [chatbot, current_conv]
            chat_send.click(
                fn=chat_stream, inputs=send_inputs, outputs=send_outputs,
            ).then(fn=lambda: "", outputs=[chat_input])

            chat_input.submit(
                fn=chat_stream, inputs=send_inputs, outputs=send_outputs,
            ).then(fn=lambda: "", outputs=[chat_input])

            demo.load(fn=refresh_conv_list, outputs=[conv_list])

        # ========== Tab 2: 文件入库 ==========
        with gr.TabItem("📤 文件入库"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 上传文件")
                    upload_files = gr.File(
                        label="选择文件",
                        file_count="multiple",
                        file_types=[".pdf", ".doc", ".docx", ".xls", ".xlsx",
                                     ".txt", ".md", ".ppt", ".pptx"],
                    )
                    with gr.Row():
                        upload_domain = gr.Dropdown(
                            choices=["自动", "变电", "配电", "送电输电", "综合"],
                            value="自动", label="域",
                        )
                        upload_category = gr.Textbox(
                            label="类目", placeholder="如: 标准规范",
                        )
                    upload_btn = gr.Button("上传并入库", variant="primary")
                    upload_result = gr.HTML(
                        value=render_empty_state("等待上传...", "选择文件后点击「上传并入库」"),
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### 入库说明")
                    gr.Markdown("""
                    <div style="font-size:13px;color:#4b5563;line-height:1.8;">
                    <p>✅ 支持 <b>PDF / DOC / DOCX / XLS / XLSX / PPT / PPTX / TXT / MD / OFD</b></p>
                    <p>✅ 可多选文件批量上传</p>
                    <p>✅ 域和类目为空时自动识别</p>
                    <p>✅ 已入库文件自动跳过</p>
                    <p>⏱ 单个大文件 OCR 可能耗时数分钟</p>
                    </div>
                    """)

            upload_btn.click(
                fn=upload_and_index,
                inputs=[upload_files, upload_domain, upload_category],
                outputs=[upload_result],
            )

        # ========== Tab 3: 文件管理 ==========
        tab_file_mgmt = gr.TabItem("📂 文件管理")
        with tab_file_mgmt:
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        file_status_filter = gr.Dropdown(
                            choices=["全部", "completed", "failed", "deleted"],
                            value="全部", label="状态", scale=1,
                        )
                        file_domain_filter = gr.Dropdown(
                            choices=["全部", "变电", "配电", "送电输电", "综合"],
                            value="全部", label="域", scale=1,
                        )
                        file_search = gr.Textbox(
                            label="搜索", placeholder="文件名或编号", scale=2,
                        )
                    with gr.Row():
                        refresh_btn = gr.Button("刷新列表", variant="primary")
                    file_tree = gr.HTML(
                        value=render_empty_state("点击刷新查看文件..."),
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 文件操作")
                    file_selector = gr.Dropdown(
                        label="选择文件", choices=[], interactive=True,
                    )
                    file_detail = gr.HTML(
                        value=render_empty_state("选择文件后查看详情"),
                    )
                    with gr.Row():
                        view_btn = gr.Button("查看详情")
                        delete_btn = gr.Button("删除文件", variant="stop")
                    gr.Markdown("#### 一致性维护")
                    with gr.Row():
                        scan_orphan_btn = gr.Button("扫描失效", variant="secondary", size="sm")
                        clean_orphan_btn = gr.Button("清理失效", variant="stop", size="sm")
                    op_result = gr.HTML("")

            refresh_btn.click(
                fn=refresh_file_list,
                inputs=[file_status_filter, file_domain_filter, file_search],
                outputs=[file_tree, file_selector],
            )
            view_btn.click(
                fn=view_file_details, inputs=[file_selector], outputs=[file_detail],
            )
            delete_btn.click(
                fn=delete_selected, inputs=[file_selector],
                outputs=[op_result, file_tree, file_selector],
            )
            scan_orphan_btn.click(
                fn=lambda: sync_orphan_files(False),
                outputs=[op_result, file_tree, file_selector],
            )
            clean_orphan_btn.click(
                fn=lambda: sync_orphan_files(True),
                outputs=[op_result, file_tree, file_selector],
            )

            demo.load(
                fn=refresh_file_list,
                inputs=[file_status_filter, file_domain_filter, file_search],
                outputs=[file_tree, file_selector],
            )

        # 切换到文件管理 tab 自动刷新
        tab_file_mgmt.select(
            fn=_on_tab_select_file_mgmt, outputs=[file_tree, file_selector],
        )

        # ========== Tab 4: 文档检索 ==========
        with gr.TabItem("🔍 文档检索"):
            with gr.Row():
                with gr.Column(scale=3):
                    srch_query = gr.Textbox(
                        label="检索关键词",
                        lines=2,
                        placeholder="仅在向量库中搜索相关文档，不生成 AI 回答",
                    )
                with gr.Column(scale=1):
                    srch_domain = gr.Dropdown(
                        choices=["全部", "变电", "配电", "送电输电", "综合"],
                        value="全部", label="域过滤",
                    )
                    srch_topk = gr.Slider(3, 30, 10, 1, label="结果数")
            with gr.Row():
                srch_btn = gr.Button("检索", variant="primary")
            srch_output = gr.HTML(
                value=render_empty_state("输入关键词开始检索..."),
            )
            srch_btn.click(
                fn=search_only,
                inputs=[srch_query, srch_domain, srch_topk],
                outputs=[srch_output],
            )

    # ==== 品牌页脚 ====
    gr.HTML(value=render_footer(), elem_id="brand-footer")

    # ==== 页面加载: 刷新头部 + 恢复 pending ====
    demo.load(
        fn=auto_recover_pending,
        outputs=[recovery_banner],
    ).then(
        fn=refresh_header,
        outputs=[header_html],
    ).then(
        fn=lambda: refresh_file_list("全部", "全部", ""),
        outputs=[file_tree, file_selector],
    )

    # 切换任意 tab 时刷新头部统计
    tabs.select(
        fn=_on_tab_select_any,
        outputs=[header_html],
    )


if __name__ == "__main__":
    import os
    # 绕过 Windows 系统代理，避免代理客户端未启动时 httpx 连接失败
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1,0.0.0.0,.local"

    print("RAG UI v4.0 — Engineering Blue theme")
    print(f"  Backend: {API_BASE}")
    print("  UI:      http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
