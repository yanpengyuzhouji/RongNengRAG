"""
榕能电力审图知识库 — 自定义 Gradio 主题
工业工程风格: 工程蓝 + 冷灰底色 + 铜色强调
"""

import gradio as gr


def create_theme() -> gr.themes.Base:
    """构建自定义主题: 工程蓝主色、冷灰中性色、收紧圆角、系统字体"""

    font_stack = (
        "system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', "
        "'PingFang SC', 'Noto Sans SC', 'Helvetica Neue', sans-serif"
    )
    font_mono_stack = (
        "'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', "
        "'Source Code Pro', 'Courier New', monospace"
    )

    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
        font=font_stack,
        font_mono=font_mono_stack,
    )

    # 精细调整 — 冷灰底色 + 收紧间距
    theme.set(
        # 页面底色: 冷灰替代暖白
        body_background_fill="#f0f2f5",
        body_background_fill_dark="#0f1419",
        # 卡片/面板
        block_background_fill="#ffffff",
        block_background_fill_dark="#1a202c",
        block_border_color="#e2e5ea",
        block_border_color_dark="#2d3643",
        block_border_width="1px",
        block_radius="0",
        block_padding="16px",
        # 面板
        panel_background_fill="#ffffff",
        panel_background_fill_dark="#1a202c",
        panel_border_color="#e2e5ea",
        panel_border_color_dark="#2d3643",
        panel_border_width="1px",
        # 输入框
        input_background_fill="#ffffff",
        input_background_fill_dark="#1e2632",
        input_border_color="#d1d5db",
        input_border_color_dark="#374151",
        input_border_color_focus="#1a56db",
        input_border_color_focus_dark="#3b82f6",
        input_border_width="1px",
        input_radius="5px",
        input_padding="10px 12px",
        # 按钮
        button_primary_background_fill="#1a56db",
        button_primary_background_fill_hover="#1e40af",
        button_primary_background_fill_dark="#2563eb",
        button_primary_background_fill_hover_dark="#3b82f6",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="#1a56db",
        button_primary_border_color_dark="#2563eb",
        button_secondary_background_fill="#f3f4f6",
        button_secondary_background_fill_hover="#e5e7eb",
        button_secondary_border_color="#d1d5db",
        button_cancel_background_fill="#ffffff",
        button_cancel_background_fill_hover="#fef2f2",
        button_cancel_border_color="#fca5a5",
        button_cancel_text_color="#dc2626",
        button_small_radius="4px",
        button_medium_radius="5px",
        button_large_radius="6px",
        # 链接
        link_text_color="#1a56db",
        link_text_color_hover="#1e40af",
        link_text_color_dark="#60a5fa",
        # 文字
        body_text_color="#1f2937",
        body_text_color_dark="#e5e7eb",
        body_text_color_subdued="#6b7280",
        body_text_color_subdued_dark="#9ca3af",
        body_text_size="14px",
        # 间距
        layout_gap="16px",
        form_gap_width="12px",
        # 强调色
        color_accent="#1a56db",
        color_accent_soft="#dbeafe",
        color_accent_soft_dark="#1e3a5f",
        # 圆角统一收紧
        container_radius="6px",
        embed_radius="6px",
        # Chatbot
        chatbot_text_size="14px",
        # 错误状态
        error_background_fill="#fef2f2",
        error_border_color="#fecaca",
        error_text_color="#dc2626",
        error_icon_color="#dc2626",
        # 加载动画
        loader_color="#1a56db",
        loader_color_dark="#3b82f6",
        # 表格
        table_border_color="#e5e7eb",
        table_even_background_fill="#f9fafb",
        table_odd_background_fill="#ffffff",
        table_radius="6px",
        # Tab
        block_title_text_size="15px",
        block_title_text_weight="600",
    )

    return theme
