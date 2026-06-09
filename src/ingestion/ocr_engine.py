"""
PaddleOCR 引擎 — 子进程隔离模式

通过子进程运行 PaddleOCR，解决 paddlepaddle 与 pymilvus 的 protobuf 版本冲突:
  - 主进程: protobuf >= 5.27.2 (pymilvus 要求)
  - OCR 子进程: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (兼容旧 protobuf 生成代码)

用法:
    # 命令行模式 (子进程)
    python ocr_engine.py /path/to/image.png

    # API 模式 (主进程调用)
    engine = OCREngine()
    text = engine.ocr_page("/path/to/page.png")

输出格式 (stdout JSON):
    {
      "success": true,
      "text": "识别到的文本内容...",
      "lines": [{"text": "...", "confidence": 0.98, "bbox": [x1,y1,x2,y2,x3,y3,x4,y4]}],
      "elapsed_ms": 1234
    }
"""

import os
import sys
import json
import time
import io
from pathlib import Path
from typing import Optional, List, Dict


def _run_ocr_on_image(image_path: str, lang: str = "ch") -> dict:
    """
    对单张图片执行 OCR，由子进程调用。

    注意: 必须在 import paddle 之前导入 torch，
    否则 paddle 修改的 PATH 会导致 torch DLL 加载失败。
    """
    # 先导入 torch (固定其 DLL 搜索路径)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    # 再导入 paddle / paddleocr
    from paddleocr import PaddleOCR

    if not os.path.exists(image_path):
        return {"success": False, "error": f"文件不存在: {image_path}"}

    t0 = time.time()

    try:
        ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
        result = ocr.ocr(image_path)

        lines = []
        all_text = []

        if result and result[0]:
            for line in result[0]:
                box, (text, confidence) = line
                # box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                flat_box = [round(v) for point in box for v in point]
                lines.append({
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "bbox": flat_box,
                })
                all_text.append(text)

        elapsed = (time.time() - t0) * 1000
        return {
            "success": True,
            "text": "\n".join(all_text),
            "lines": lines,
            "elapsed_ms": round(elapsed, 1),
        }

    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return {
            "success": False,
            "error": str(e),
            "elapsed_ms": round(elapsed, 1),
        }


def _run_ocr_on_pdf_pages(pdf_path: str, pages: List[int],
                          dpi: int = 200, lang: str = "ch") -> dict:
    """
    对 PDF 指定页面执行 OCR。
    先逐页渲染为 PNG，再逐页识别。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"success": False, "error": "PyMuPDF (fitz) 未安装"}

    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    from paddleocr import PaddleOCR

    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"PDF 不存在: {pdf_path}"}

    t0 = time.time()
    doc = fitz.open(pdf_path)
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)

    pages_result = {}
    all_text_parts = []
    total_pages = 0

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="ocr_pages_")

    try:
        for page_num in pages:
            if page_num < 0 or page_num >= len(doc):
                continue

            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img_path = os.path.join(tmpdir, f"page_{page_num + 1}.png")
            pix.save(img_path)

            result = ocr.ocr(img_path)

            page_lines = []
            if result and result[0]:
                for line in result[0]:
                    box, (text, confidence) = line
                    page_lines.append({
                        "text": text,
                        "confidence": round(float(confidence), 4),
                    })
                    all_text_parts.append(text)

            pages_result[str(page_num + 1)] = {
                "lines": page_lines,
                "text": "\n".join([l["text"] for l in page_lines]),
            }
            total_pages += 1

            # 清理临时图片
            try:
                os.remove(img_path)
            except OSError:
                pass

    finally:
        doc.close()
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    elapsed = (time.time() - t0) * 1000
    return {
        "success": True,
        "text": "\n".join(all_text_parts),
        "pages": pages_result,
        "total_pages_ocr": total_pages,
        "elapsed_ms": round(elapsed, 1),
    }


def _stdout_json(obj: dict):
    """输出 JSON 到 stdout (子进程通信)"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    json.dump(obj, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()


def _run_ocr_batch(image_paths: List[str], lang: str = "ch") -> List[dict]:
    """
    批量 OCR 多张图片 — 一次性加载模型，处理所有图片。

    用于子进程批量模式: 避免每张图片都重新加载 PaddleOCR 模型。
    """
    # 先导入 torch (固定 DLL)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    from paddleocr import PaddleOCR

    t0 = time.time()
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)

    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            results.append({
                "success": False,
                "error": f"文件不存在: {img_path}",
                "lines": [],
                "text": "",
            })
            continue

        t_img = time.time()
        try:
            raw = ocr.ocr(img_path)
            lines = []
            all_text = []
            if raw and raw[0]:
                for line in raw[0]:
                    box, (text, confidence) = line
                    flat_box = [round(v) for point in box for v in point]
                    lines.append({
                        "text": text,
                        "confidence": round(float(confidence), 4),
                        "bbox": flat_box,
                    })
                    all_text.append(text)

            results.append({
                "success": True,
                "text": "\n".join(all_text),
                "lines": lines,
                "elapsed_ms": round((time.time() - t_img) * 1000, 1),
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "lines": [],
                "text": "",
                "elapsed_ms": round((time.time() - t_img) * 1000, 1),
            })

    return results


# ===== CLI 入口 (子进程模式) =====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        result = {"success": False, "error": "Usage: python ocr_engine.py <image_or_pdf_path> [--pages 1,2,3] [--batch]"}
        _stdout_json(result)
        sys.exit(1)

    # --batch 模式: 从 stdin 读取 JSON 图片路径列表, 批量 OCR
    if "--batch" in sys.argv:
        raw = sys.stdin.read()
        try:
            image_paths = json.loads(raw)
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
        except json.JSONDecodeError:
            result = [{"success": False, "error": "stdin 不是有效 JSON"}]
            _stdout_json(result)
            sys.exit(1)

        results = _run_ocr_batch(image_paths)
        _stdout_json(results)
        sys.exit(0)

    input_path = sys.argv[1]
    pages = None

    # 解析 --pages 参数
    if "--pages" in sys.argv:
        idx = sys.argv.index("--pages")
        if idx + 1 < len(sys.argv):
            pages = [int(p.strip()) - 1 for p in sys.argv[idx + 1].split(",")]

    if pages:
        # PDF 多页 OCR (单进程内完成, 不通过 CLI)
        result = _run_ocr_on_pdf_pages(input_path, pages)
    elif input_path.lower().endswith('.pdf') and not pages:
        result = {"success": False, "error": "PDF 需要指定 --pages 参数"}
    else:
        # 单张图片 OCR
        result = _run_ocr_on_image(input_path)

    _stdout_json(result)


# ===== Python API (主进程调用) =====
class OCREngine:
    """
    OCR 引擎封装 — 主进程通过 subprocess 调用 OCR 子进程。

    用法:
        engine = OCREngine()
        text = engine.ocr_page("/path/to/page.png")
        # 或
        pdf_text = engine.ocr_pdf_pages("/path/to/doc.pdf", [0, 1, 4])

    首次调用会较慢（子进程启动 + 模型加载），后续调用复用子进程。
    """

    def __init__(self, lang: str = "ch", dpi: int = 200,
                 use_subprocess: bool = True):
        """
        Args:
            lang: OCR 语言 ("ch" = 中文)
            dpi: 渲染 PDF 页面的 DPI (越大越清晰，越慢)
            use_subprocess: 是否使用子进程隔离 (默认 True)
        """
        self.lang = lang
        self.dpi = dpi
        self.use_subprocess = use_subprocess
        self._ocr_instance = None  # 直接模式下的缓存

    def _get_ocr(self):
        """获取 OCR 实例（直接模式，需 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python）"""
        if self._ocr_instance is None:
            from paddleocr import PaddleOCR
            self._ocr_instance = PaddleOCR(
                lang=self.lang, use_angle_cls=True, show_log=False
            )
        return self._ocr_instance

    def ocr_page(self, image_path: str) -> dict:
        """
        OCR 识别单张图片

        Args:
            image_path: 图片文件路径 (PNG/JPEG 等)

        Returns:
            {"success": bool, "text": str, "lines": [...], "elapsed_ms": float}
        """
        if self.use_subprocess:
            return self._ocr_subprocess(image_path)

        # 直接模式
        if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        return _run_ocr_on_image(image_path, self.lang)

    def ocr_pdf_pages(self, pdf_path: str,
                      pages: List[int],
                      tmp_dir: str = None) -> dict:
        """
        对 PDF 指定页面执行 OCR

        工作流程:
        1. 用 PyMuPDF 将每个页面渲染为 PNG (主进程)
        2. 调用 OCR 子进程识别每张图片
        3. 返回合并结果

        Args:
            pdf_path: PDF 文件路径
            pages: 需要 OCR 的页码列表 (0-based)
            tmp_dir: 临时图片目录 (默认系统临时目录)

        Returns:
            {"success": bool, "text": str, "pages": {...}, "elapsed_ms": float}
        """
        if not pages:
            return {"success": True, "text": "", "pages": {}, "total_pages_ocr": 0, "elapsed_ms": 0}

        if self.use_subprocess:
            return self._ocr_pdf_subprocess(pdf_path, pages, tmp_dir)

        # 直接模式
        if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        return _run_ocr_on_pdf_pages(pdf_path, pages, self.dpi, self.lang)

    def _ocr_subprocess(self, image_path: str) -> dict:
        """通过子进程调用 OCR (单图)"""
        import subprocess
        import tempfile

        script = Path(__file__).resolve()

        env = os.environ.copy()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["PYTHONIOENCODING"] = "utf-8"
        # 抑制 paddle 的 verbose 日志
        env["GLOG_minloglevel"] = "2"

        try:
            proc = subprocess.run(
                [sys.executable, str(script), image_path],
                capture_output=True, encoding="utf-8", timeout=120,
                env=env,
                cwd=str(script.parent.parent.parent),  # 项目根目录
            )

            if proc.returncode != 0:
                stderr = proc.stderr or ""
                return {
                    "success": False,
                    "error": f"OCR 子进程退出码 {proc.returncode}: {stderr[:500]}"
                }

            return json.loads(proc.stdout)

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "OCR 子进程超时 (120s)"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"OCR 子进程返回无效 JSON: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ocr_pdf_subprocess(self, pdf_path: str, pages: List[int],
                            tmp_dir: str = None) -> dict:
        """通过子进程调用 OCR (PDF 多页，批量模式)

        将所有页面渲染为 PNG 后，在一个子进程中批量 OCR，
        避免每页启动子进程的开销（每页省 10-15s 模型加载时间）。
        """
        import subprocess
        import tempfile
        import fitz  # PyMuPDF

        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
            cleanup_dir = tmp_dir
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            cleanup_dir = None

        script = Path(__file__).resolve()
        env = os.environ.copy()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["PYTHONIOENCODING"] = "utf-8"
        env["GLOG_minloglevel"] = "2"

        t0 = time.time()
        doc = fitz.open(pdf_path)
        image_paths = []
        pages_result = {}
        all_text_parts = []
        total_pages = 0

        try:
            # Step 1: 渲染所有需要 OCR 的页面为 PNG
            for page_num in pages:
                if page_num < 0 or page_num >= len(doc):
                    continue
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.dpi)
                img_path = os.path.join(tmp_dir, f"page_{page_num + 1}.png")
                pix.save(img_path)
                image_paths.append((page_num, img_path))

            # Step 2: 在一个子进程中批量 OCR 所有图片
            # 将图片路径通过 stdin 传给子进程
            img_list_json = json.dumps(
                [p for _, p in image_paths], ensure_ascii=False
            )

            try:
                proc = subprocess.run(
                    [sys.executable, str(script), "--batch"],
                    input=img_list_json,
                    capture_output=True, encoding="utf-8", timeout=600,
                    env=env,
                    cwd=str(script.parent.parent.parent),
                )

                if proc.returncode == 0 and proc.stdout.strip():
                    batch_results = json.loads(proc.stdout)
                else:
                    stderr = proc.stderr or ""
                    # 批量失败，回退到逐页 OCR
                    print(f"   [OCR] 批量 OCR 失败，回退逐页: {stderr[:200]}")
                    batch_results = None
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                print(f"   [OCR] 批量 OCR 异常: {e}")
                batch_results = None

            # Step 3: 解析结果
            if batch_results and isinstance(batch_results, list):
                for i, (page_num, _) in enumerate(image_paths):
                    if i < len(batch_results):
                        page_result = batch_results[i]
                        if page_result.get("success"):
                            pages_result[str(page_num + 1)] = {
                                "lines": page_result.get("lines", []),
                                "text": page_result.get("text", ""),
                            }
                            all_text_parts.append(
                                page_result.get("text", "")
                            )
                    total_pages += 1
            else:
                # 回退：逐页 OCR
                for page_num, img_path in image_paths:
                    try:
                        proc = subprocess.run(
                            [sys.executable, str(script), img_path],
                            capture_output=True, encoding="utf-8",
                            timeout=120, env=env,
                            cwd=str(script.parent.parent.parent),
                        )
                        if proc.returncode == 0:
                            page_result = json.loads(proc.stdout)
                        else:
                            page_result = {
                                "success": False,
                                "error": f"子进程退出码 {proc.returncode}"
                            }
                    except subprocess.TimeoutExpired:
                        page_result = {"success": False, "error": "OCR 超时"}
                    except json.JSONDecodeError:
                        page_result = {"success": False, "error": "OCR 返回无效 JSON"}

                    if page_result.get("success"):
                        pages_result[str(page_num + 1)] = {
                            "lines": page_result.get("lines", []),
                            "text": page_result.get("text", ""),
                        }
                        all_text_parts.append(page_result.get("text", ""))
                    total_pages += 1

        finally:
            doc.close()
            # 清理临时文件
            for _, img_path in image_paths:
                try:
                    os.remove(img_path)
                except OSError:
                    pass
            if cleanup_dir:
                try:
                    os.rmdir(cleanup_dir)
                except OSError:
                    pass

        elapsed = (time.time() - t0) * 1000

        # 统计有效页
        valid_pages = sum(
            1 for v in pages_result.values()
            if v.get("text", "").strip()
        )
        if valid_pages > 0:
            print(f"   [OCR] {valid_pages}/{total_pages} 页有效识别, "
                  f"耗时 {elapsed:.0f}ms ({elapsed / total_pages:.0f}ms/页)")

        return {
            "success": True,
            "text": "\n".join(all_text_parts),
            "pages": pages_result,
            "total_pages_ocr": total_pages,
            "elapsed_ms": round(elapsed, 1),
        }
