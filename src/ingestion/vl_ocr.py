"""
PaddleOCR-VL 结构化识别客户端 — 调用外部 OpenAI 兼容的 VL 模型服务
将 PDF 页面渲染为图片 → base64 → POST /v1/chat/completions → 返回结构化文本
支持阅读顺序、markdown 表格、标题层级 (PaddleOCR-VL 1.x 特性)
"""

import base64
import io
import time
from typing import Dict, List, Optional

import fitz  # PyMuPDF


# 识别提示词: 要求保持阅读顺序 + 结构化输出
_VL_PROMPT = (
    "识别图中全部文字，保持阅读顺序输出；"
    "表格转为 markdown 表格；保留标题层级；"
    "只输出识别到的内容，不要额外解释。"
)


class VLOcrClient:
    """调用外部 PaddleOCR-VL 服务 (OpenAI 兼容协议) 进行结构化 OCR"""

    def __init__(self, base_url: str, model: str = "paddleocr-vl-1.6",
                 timeout: int = 180, dpi: int = 150, max_image_dim: int = 3000):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dpi = dpi
        self.max_image_dim = max_image_dim
        self._client = None
        self._ready = False
        self._ensure_client()

    def _ensure_client(self):
        """延迟初始化 OpenAI 客户端 (线程安全由调用方保证)"""
        if self._ready:
            return
        from openai import OpenAI
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="sk-empty",  # 本地服务通常不校验 key
            timeout=self.timeout,
        )
        self._ready = True
        print(f"[vl-ocr] PaddleOCR-VL 服务: {self.base_url}, model={self.model}")

    def _render_page_to_png(self, page: fitz.Page) -> bytes:
        """将 PDF 页面渲染为 PNG bytes"""
        pix = page.get_pixmap(dpi=self.dpi)

        # 大图缩放 (限制最大边长, 控制请求体积)
        if self.max_image_dim > 0 and max(pix.width, pix.height) > self.max_image_dim:
            from PIL import Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img.thumbnail((self.max_image_dim, self.max_image_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return pix.tobytes("png")

    def recognize_image(self, image_bytes: bytes) -> str:
        """识别单张图片, 返回结构化文本 (markdown)"""
        self._ensure_client()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VL_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()

    def recognize_pdf_pages(self, pdf_path: str,
                            pages_0based: List[int]) -> Dict[int, str]:
        """
        对 PDF 指定页面执行结构化 OCR

        Args:
            pdf_path: PDF 文件路径
            pages_0based: 需要识别的页面索引 (0-based)

        Returns:
            {0-based_page_index: 结构化文本}
        """
        result: Dict[int, str] = {}
        if not pages_0based:
            return result

        doc = fitz.open(pdf_path)
        try:
            for page_idx in pages_0based:
                if page_idx >= len(doc):
                    continue
                t0 = time.time()
                try:
                    img = self._render_page_to_png(doc[page_idx])
                    text = self.recognize_image(img)
                    if text:
                        result[page_idx] = text
                    else:
                        print(f"   [vl-ocr] 第{page_idx+1}页无有效内容")
                except Exception as e:
                    print(f"   [vl-ocr] 第{page_idx+1}页识别失败: {e}")
                elapsed = time.time() - t0
                print(f"   [vl-ocr] 第{page_idx+1}页 → {len(result.get(page_idx,''))}字符 "
                      f"({elapsed:.1f}s)")
        finally:
            doc.close()
        return result
