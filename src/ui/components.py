"""
榕能电力审图知识库 — 可复用 HTML 组件
用于 gr.HTML() 嵌入，突破 Gradio 原生组件限制
"""

import json
import time
from typing import Optional


# ====== 品牌头部 ======

def render_header(
    total_files: int = 0,
    total_chunks: int = 0,
    total_chars: int = 0,
    online: bool = True,
) -> str:
    """品牌头部: Logo + 标题 + 统计摘要条"""
    status_color = "#059669" if online else "#dc2626"
    status_text = "在线" if online else "离线"
    status_dot = f"""<span style="display:inline-block;width:8px;height:8px;border-radius:50%;
        background:{status_color};margin-right:4px;box-shadow:0 0 6px {status_color};"></span>"""

    return f"""<div class="brand-header" style="
        display:flex;align-items:center;justify-content:space-between;
        padding:12px 24px;background:linear-gradient(135deg,#111827 0%,#1e293b 100%);
        margin:-16px -16px 0 -16px;
        font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;
    ">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                width:36px;height:36px;border-radius:6px;
                background:linear-gradient(135deg,#1a56db,#2563eb);
                display:flex;align-items:center;justify-content:center;
                font-size:20px;
            ">⚡</div>
            <div>
                <div style="color:#f9fafb;font-size:15px;font-weight:600;letter-spacing:0.5px;">
                    榕能电力审图知识库
                </div>
                <div style="color:#9ca3af;font-size:11px;font-weight:400;">
                    RAG v3.0 · 智能问答系统
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:18px;">
            <div class="header-stat" style="text-align:center;">
                <div style="color:#9ca3af;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">文档</div>
                <div style="color:#f9fafb;font-size:18px;font-weight:700;font-variant-numeric:tabular-nums;">{total_files}</div>
            </div>
            <div class="header-stat" style="text-align:center;">
                <div style="color:#9ca3af;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">Chunks</div>
                <div style="color:#f9fafb;font-size:18px;font-weight:700;font-variant-numeric:tabular-nums;">{total_chunks:,}</div>
            </div>
            <div style="width:1px;height:28px;background:#374151;"></div>
            <div style="display:flex;align-items:center;gap:6px;color:#d1d5db;font-size:12px;">
                {status_dot} {status_text}
            </div>
        </div>
    </div>"""


# ====== 品牌页脚 ======

def render_footer() -> str:
    """品牌页脚: 技术栈 + 北京时间"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return f"""<div style="
        text-align:center;padding:12px 24px;margin:0 -16px;
        font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;
    ">
        <span style="font-weight:600;color:#6b7280;">Milvus Lite</span>
        <span style="margin:0 6px;color:#d1d5db;">·</span>
        <span style="font-weight:600;color:#6b7280;">BGE-M3</span>
        <span style="margin:0 6px;color:#d1d5db;">·</span>
        <span style="font-weight:600;color:#6b7280;">Qwen3.5 4B</span>
        <span style="margin:0 8px;color:#d1d5db;">|</span>
        北京时间 {ts}
    </div>"""


# ====== 状态指示器 ======

def status_dot(status: str) -> str:
    """彩色状态圆点"""
    colors = {
        "completed": "#059669",
        "processing": "#d97706",
        "pending": "#d97706",
        "failed": "#dc2626",
        "deleted": "#9ca3af",
    }
    labels = {
        "completed": "完成",
        "processing": "处理中",
        "pending": "待处理",
        "failed": "失败",
        "deleted": "已删除",
    }
    c = colors.get(status, "#9ca3af")
    label = labels.get(status, status)
    return f"""<span style="display:inline-flex;align-items:center;gap:4px;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{c};"></span>
        <span style="font-size:12px;color:#6b7280;">{label}</span>
    </span>"""


def domain_badge(domain: str) -> str:
    """域分类彩色标签"""
    colors = {
        "变电": ("#dbeafe", "#1e40af"),
        "配电": ("#fef3c7", "#92400e"),
        "送电输电": ("#dcfce7", "#166534"),
        "综合": ("#f3e8ff", "#6b21a8"),
    }
    bg, fg = colors.get(domain, ("#f3f4f6", "#374151"))
    return f"""<span style="display:inline-block;padding:1px 8px;border-radius:10px;
        font-size:11px;font-weight:600;background:{bg};color:{fg};">{domain}</span>"""


# ====== 文件树 (HTML 版) ======

def render_file_tree(files: list) -> str:
    """
    结构化 HTML 文件树 — 域分组折叠 + 彩色状态点 + 右对齐
    替代原有的纯 Markdown 渲染
    """
    if not files:
        return _empty_state("暂无入库文件", "请前往「文件入库」上传文档")

    # 按域和类目分组
    by_domain: dict[str, dict[str, list[dict]]] = {}
    for f in sorted(files, key=lambda x: (x.get("domain", "") or "未分类",
                                            x.get("category", "") or "未分类",
                                            x.get("file_name", ""))):
        dm = f.get("domain", "") or "未分类"
        cat = f.get("category", "") or "未分类"
        by_domain.setdefault(dm, {}).setdefault(cat, []).append(f)

    total_files = len(files)
    total_chunks = sum(f.get("chunks_count", 0) for f in files)
    missing_count = sum(1 for f in files if f.get("file_exists") is False)

    domain_colors = {
        "变电": ("#dbeafe", "#1e40af", "#3b82f6"),
        "配电": ("#fef3c7", "#92400e", "#f59e0b"),
        "送电输电": ("#dcfce7", "#166534", "#22c55e"),
        "综合": ("#f3e8ff", "#6b21a8", "#a855f7"),
        "未分类": ("#f3f4f6", "#374151", "#9ca3af"),
    }

    lines = [
        '<div class="file-tree" style="font-family:system-ui,\'Microsoft YaHei\',sans-serif;font-size:13px;">',
        # 摘要行
        f"""<div style="display:flex;align-items:center;gap:16px;padding:8px 0 12px;border-bottom:1px solid #e5e7eb;margin-bottom:12px;">
            <div><span style="font-size:18px;font-weight:700;color:#1f2937;">{total_files}</span>
                <span style="font-size:12px;color:#6b7280;"> 个文件</span></div>
            <div><span style="font-size:18px;font-weight:700;color:#1f2937;">{total_chunks:,}</span>
                <span style="font-size:12px;color:#6b7280;"> chunks</span></div>
            {f'<div style="color:#d97706;font-size:12px;">⚠ {missing_count} 个文件丢失</div>' if missing_count > 0 else ''}
        </div>""",
    ]

    for dm in sorted(by_domain.keys()):
        cats = by_domain[dm]
        dm_count = sum(len(v) for v in cats.values())
        bg_c, fg_c, accent_c = domain_colors.get(dm, ("#f3f4f6", "#374151", "#9ca3af"))

        # 域标题 (可点击折叠)
        dm_id = f"domain-{hash(dm) & 0x7FFFFFFF}"
        lines.append(f"""
        <div style="margin-bottom:8px;">
            <div class="tree-domain-header" onclick="
                var el=document.getElementById('{dm_id}');
                var icon=document.getElementById('{dm_id}-icon');
                if(el.style.display==='none'){{el.style.display='block';icon.textContent='▼';}}
                else{{el.style.display='none';icon.textContent='▶';}}
            " style="display:flex;align-items:center;gap:8px;padding:6px 10px;
                background:{bg_c};border-radius:4px;cursor:pointer;user-select:none;">
                <span id="{dm_id}-icon" style="font-size:10px;color:{fg_c};">▼</span>
                <span style="font-weight:600;color:{fg_c};">{dm}</span>
                <span style="
                    display:inline-flex;align-items:center;justify-content:center;
                    min-width:20px;height:18px;padding:0 6px;border-radius:9px;
                    font-size:11px;font-weight:600;color:#fff;background:{accent_c};">{dm_count}</span>
            </div>
            <div id="{dm_id}" style="display:block;padding-left:8px;">""")

        for cat in sorted(cats.keys()):
            cat_files = cats[cat]
            lines.append(f"""
                <div style="margin:4px 0;padding:2px 8px;">
                    <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:2px;">
                        {cat}
                        <span style="font-weight:400;color:#9ca3af;">({len(cat_files)})</span>
                    </div>""")

            for f in cat_files:
                st = f.get("status", "?")
                is_missing = f.get("file_exists") is False
                dot_c = {"completed": "#059669", "processing": "#d97706",
                         "failed": "#dc2626", "deleted": "#9ca3af"}.get(st, "#9ca3af")
                if is_missing:
                    dot_c = "#9ca3af"

                fname = f.get("file_name", "?")[:55]
                chunks = f.get("chunks_count", 0)
                created = (f.get("created_at", "") or "")[:10]
                missing_note = ' <span style="color:#9ca3af;font-size:11px;">[丢失]</span>' if is_missing else ""

                lines.append(f"""
                    <div style="display:flex;align-items:center;gap:6px;padding:2px 0;font-size:12px;
                        color:#4b5563;border-bottom:1px solid #f9fafb;">
                        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                            background:{dot_c};flex-shrink:0;"></span>
                        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                            title="{f.get('file_name','?')}">{fname}{missing_note}</span>
                        <span style="font-size:10px;color:#9ca3af;flex-shrink:0;">{chunks}c</span>
                        <span style="font-size:10px;color:#9ca3af;flex-shrink:0;width:70px;text-align:right;">{created}</span>
                    </div>""")

            lines.append("</div>")  # end category

        lines.append("</div></div>")  # end domain collapsible

    lines.append("</div>")  # end file-tree
    return "\n".join(lines)


# ====== 文件详情卡片 ======

def render_file_detail(file_data: dict) -> str:
    """文件详情 — HTML 卡片，非 Markdown 表格"""
    status = file_data.get("status", "?")
    status_info = {
        "completed": ("#059669", "已完成"),
        "processing": ("#d97706", "处理中"),
        "failed": ("#dc2626", "失败"),
        "deleted": ("#9ca3af", "已删除"),
    }
    st_color, st_label = status_info.get(status, ("#9ca3af", status))

    fields = [
        ("文件名", file_data.get("file_name", "?")),
        ("域 / 类目", f"{file_data.get('domain','-')} / {file_data.get('category','-')}"),
        ("文档编号", file_data.get("doc_number", "-")),
        ("文件大小", f"{file_data.get('file_size', 0):,} bytes"),
        ("Chunks", str(file_data.get("chunks_count", 0))),
        ("Hash", f"<code style='font-size:11px;'>{file_data.get('file_hash','?')[:24]}...</code>"),
        ("入库时间", file_data.get("created_at", "-")),
        ("原始路径", f"<code style='font-size:11px;'>{file_data.get('original_path','?')}</code>"),
    ]

    rows = ""
    for label, value in fields:
        rows += f"""<div style="display:flex;padding:6px 0;border-bottom:1px solid #f3f4f6;">
            <span style="width:80px;flex-shrink:0;font-size:12px;color:#6b7280;font-weight:500;">{label}</span>
            <span style="font-size:13px;color:#1f2937;word-break:break-all;">{value}</span>
        </div>"""

    return f"""<div style="font-family:system-ui,'Microsoft YaHei',sans-serif;font-size:13px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <span style="font-size:15px;font-weight:700;color:#1f2937;">文件详情</span>
            <span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:2px 8px;
                border-radius:10px;background:{st_color}15;color:{st_color};">
                <span style="width:6px;height:6px;border-radius:50%;background:{st_color};"></span>
                {st_label}
            </span>
        </div>
        {rows}
    </div>"""


# ====== 状态横幅 (错误/成功/警告) ======

def render_alert(message: str, level: str = "info") -> str:
    """统一的状态横幅: info / success / warning / error"""
    config = {
        "info":    ("#dbeafe", "#1e40af", "#3b82f6", "ℹ"),
        "success": ("#dcfce7", "#166534", "#22c55e", "✓"),
        "warning": ("#fef3c7", "#92400e", "#f59e0b", "⚠"),
        "error":   ("#fef2f2", "#991b1b", "#ef4444", "✕"),
    }
    bg, fg, border, icon = config.get(level, config["info"])
    return f"""<div style="display:flex;align-items:flex-start;gap:10px;
        padding:10px 14px;background:{bg};border-left:3px solid {border};
        border-radius:4px;margin:8px 0;font-family:system-ui,'Microsoft YaHei',sans-serif;">
        <span style="color:{border};font-weight:700;font-size:14px;flex-shrink:0;line-height:1.5;">{icon}</span>
        <span style="font-size:13px;color:{fg};line-height:1.5;">{message}</span>
    </div>"""


def render_empty_state(message: str, action: str = "") -> str:
    """空状态引导"""
    action_html = ""
    if action:
        action_html = f"""<div style="margin-top:8px;font-size:12px;color:#6b7280;">{action}</div>"""
    return _empty_state(message, action)


def _empty_state(message: str, action: str = "") -> str:
    """空状态占位"""
    action_html = ""
    if action:
        action_html = f"""<div style="margin-top:8px;font-size:12px;color:#6b7280;">{action}</div>"""
    return f"""<div style="text-align:center;padding:32px 16px;color:#9ca3af;
        font-family:system-ui,'Microsoft YaHei',sans-serif;">
        <div style="font-size:36px;margin-bottom:8px;">📭</div>
        <div style="font-size:13px;">{message}</div>
        {action_html}
    </div>"""


# ====== 检索结果卡 ======

def render_search_results(query: str, result_data: dict, elapsed_ms: float) -> str:
    """检索结果 HTML 卡片"""
    domain_colors = {
        "变电": ("#dbeafe", "#1e40af"),
        "配电": ("#fef3c7", "#92400e"),
        "送电输电": ("#dcfce7", "#166534"),
        "综合": ("#f3e8ff", "#6b21a8"),
    }

    results = result_data.get("results", [])
    total = result_data.get("total_candidates", 0)
    qtype = result_data.get("query_type", "?")

    if not results:
        return render_empty_state("未检索到相关文档", "尝试调整域过滤或使用更精确的关键词")

    lines = [
        '<div style="font-family:system-ui,\'Microsoft YaHei\',sans-serif;">',
        f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;padding-bottom:8px;
            border-bottom:1px solid #e5e7eb;">
            <span style="font-size:14px;font-weight:600;color:#1f2937;">检索: </span>
            <span style="font-size:14px;color:#374151;">{_escape(query)}</span>
            <span style="font-size:11px;color:#6b7280;margin-left:auto;">
                {qtype} · {total} 候选 · {elapsed_ms:.0f}ms</span>
        </div>""",
    ]

    for i, item in enumerate(results):
        dm = item.get("domain", "")
        bg, fg = domain_colors.get(dm, ("#f3f4f6", "#374151"))
        fname = item.get("file_path", "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if not fname:
            fname = item.get("doc_number", "") or "未知文件"
        confidence = item.get("confidence", 0)
        conf_pct = f"{confidence:.0%}"
        text_snippet = (item.get("text", "") or "")[:280]

        bar_fill = min(int(confidence * 100), 100)
        bar_color = "#059669" if confidence >= 0.7 else ("#d97706" if confidence >= 0.4 else "#dc2626")

        lines.append(f"""
        <div style="margin-bottom:14px;padding:12px;background:#fff;border:1px solid #e5e7eb;
            border-radius:6px;border-left:3px solid {bar_color};">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;
                    font-weight:600;background:{bg};color:{fg};">{dm}</span>
                <span style="font-size:13px;font-weight:600;color:#1f2937;">[{i+1}] {_escape(fname)}</span>
                <span style="margin-left:auto;font-size:11px;color:#6b7280;">
                    {dm}/{item.get('category','-')}
                    {f"| {item.get('voltage_level')}" if item.get('voltage_level') else ''}
                </span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:11px;font-weight:600;color:{bar_color};">{conf_pct}</span>
                <div style="flex:1;height:4px;background:#f3f4f6;border-radius:2px;">
                    <div style="width:{bar_fill}%;height:100%;background:{bar_color};border-radius:2px;"></div>
                </div>
            </div>
            <div style="font-size:12px;color:#4b5563;line-height:1.6;padding:8px;background:#f9fafb;
                border-radius:4px;">{_escape(text_snippet)}</div>
        </div>""")

    lines.append("</div>")
    return "\n".join(lines)


# ====== 入库进度 ======

def render_upload_progress(
    current: int, total: int, results: list, ok_count: int = 0, done: bool = False
) -> str:
    """入库进度 HTML — 进度条 + 实时文件表格
    results: list of (icon, filename, chunks, chars, time_str) tuples
    """
    pct = int(current / max(total, 1) * 100)

    header = "入库完成" if done else f"入库进度 ({current}/{total})"
    header_icon = "✅" if done else "📤"

    lines = [
        '<div style="font-family:system-ui,\'Microsoft YaHei\',sans-serif;">',
        f"""<div style="margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:16px;">{header_icon}</span>
                <span style="font-size:15px;font-weight:700;color:#1f2937;">{header}</span>
                {f'<span style="font-size:13px;color:#059669;margin-left:auto;font-weight:600;">{ok_count}/{total} 成功</span>' if done else ''}
            </div>
            <div style="height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#1a56db,#3b82f6);
                    border-radius:3px;transition:width 0.3s ease;"></div>
            </div>
            <div style="text-align:right;font-size:11px;color:#6b7280;margin-top:2px;">{pct}%</div>
        </div>""",
    ]

    if results:
        lines.append("""<div style="max-height:320px;overflow-y:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr style="background:#f9fafb;text-align:left;position:sticky;top:0;">
                <th style="padding:6px 8px;width:32px;"></th>
                <th style="padding:6px 8px;">文件</th>
                <th style="padding:6px 8px;text-align:right;">Chunks</th>
                <th style="padding:6px 8px;text-align:right;">字符</th>
                <th style="padding:6px 8px;text-align:right;">耗时</th>
            </tr>""")

        for icon, fname, chunks, chars, t in results:
            lines.append(
                f"<tr style='border-bottom:1px solid #f3f4f6;'>"
                f"<td style='padding:4px 8px;'>{icon}</td>"
                f"<td style='padding:4px 8px;max-width:200px;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap;' title='{_escape(str(fname))}'>"
                f"{_escape(str(fname)[:50])}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{chunks}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{chars}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{t}</td>"
                f"</tr>"
            )

        lines.append("</table></div>")

    lines.append("</div>")
    return "\n".join(lines)


# ====== 辅助 ======

def _escape(s: str) -> str:
    """简单 HTML 转义"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
