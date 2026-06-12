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
import contextlib
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def _parse_json_stdout(stdout_text: str):
    """
    从子进程 stdout 中提取 JSON — 过滤掉 PaddleOCR 初始化日志

    PaddleOCR 初始化时往 stdout 打印模型加载信息，与 stdout_json()
    的 JSON 输出混在一起。此函数尝试多种方式提取有效 JSON。
    """
    if not stdout_text or not stdout_text.strip():
        return None
    # 方法1: 从后往前找 JSON 行 (对象 { 或 数组 [)
    lines = stdout_text.strip().split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    # 方法2: 直接解析整个 stdout
    try:
        return json.loads(stdout_text)
    except json.JSONDecodeError:
        return None


def _wait_for_gpu_slot(task_name: str = "OCR", min_free_mb: int = 2500):
    """
    GPU 模式背压: 启动 OCR 前等待显存释放

    策略:
      - 空闲显存不足 → 检查 Ollama 是否在推理
      - Ollama 繁忙 → 等待 (LLM 优先)
      - Ollama 空闲 → 等待 30s 后强制执行
    """
    try:
        # 确保 src 目录在 path 中
        src_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
        sys.path.insert(0, os.path.dirname(os.path.dirname(src_dir)))
        from src.utils.gpu_monitor import get_gpu_monitor
        monitor = get_gpu_monitor(min_free_vram_mb=min_free_mb)
        ok = monitor.wait_for_vram(min_free_mb=min_free_mb)
        if not ok:
            logger.warning(f"[{task_name}] 显存等待超时，强制执行")
    except ImportError:
        pass  # GPU 监控不可用，直接执行
    except Exception as e:
        logger.warning(f"[{task_name}] 显存检查失败: {e}，直接执行")


def _get_paddleocr_version() -> int:
    """返回 PaddleOCR 主版本号 (2 或 3)"""
    try:
        from paddleocr import __version__ as v
        return int(v.split(".")[0])
    except Exception:
        return 2  # 默认按 2.x 处理


def _build_ocr_kwargs(lang, use_angle_cls, use_gpu, cpu_threads=0):
    """
    根据 PaddleOCR 版本构建正确的构造参数。
    2.x: use_angle_cls, use_gpu, show_log, cpu_threads
    3.x: use_textline_orientation, device
    """
    ver = _get_paddleocr_version()
    if ver >= 3:
        return {
            "lang": lang,
            "use_textline_orientation": use_angle_cls,
            "device": "gpu" if use_gpu else "cpu",
        }
    else:
        return {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "use_gpu": use_gpu,
            "show_log": False,
            "cpu_threads": cpu_threads,
        }


def _limit_cpu_threads(cpu_threads: int):
    """
    设置 CPU 线程数限制，防止 OCR 占满所有核心导致过热。

    通过环境变量在 PaddlePaddle 加载前限制底层 BLAS 库的线程数。
    PaddlePaddle 本身通过 cpu_threads 参数控制推理线程数。
    """
    if cpu_threads <= 0:
        return
    threads_str = str(cpu_threads)
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = threads_str


def _run_ocr_on_image(image_path: str, lang: str = "ch",
                      cpu_threads: int = 0,
                      use_angle_cls: bool = True,
                      use_gpu: bool = False) -> dict:
    """
    对单张图片执行 OCR，由子进程调用。

    注意: 必须在 import paddle 之前导入 torch，
    否则 paddle 修改的 PATH 会导致 torch DLL 加载失败。
    """
    import contextlib

    # CPU 线程限制 (必须在加载 BLAS 库之前设置)
    _limit_cpu_threads(cpu_threads)

    # 先导入 torch (固定其 DLL 搜索路径)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    # 抑制 PaddleOCR 初始化日志 (它往 stdout 打印，会破坏 JSON)
    with contextlib.redirect_stdout(io.StringIO()):
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            **_build_ocr_kwargs(lang, use_angle_cls, use_gpu, cpu_threads),
        )

    if not os.path.exists(image_path):
        return {"success": False, "error": f"文件不存在: {image_path}"}

    t0 = time.time()

    try:
        result = ocr.ocr(image_path)

        lines = []
        all_text = []

        if result and len(result) > 0:
            r0 = result[0]
            # PaddleOCR 3.x: OCRResult dict with rec_texts/rec_scores
            if isinstance(r0, dict) and "rec_texts" in r0:
                rec_texts = r0.get("rec_texts", [])
                rec_scores = r0.get("rec_scores", []) or [0.0] * len(rec_texts)
                for text, score in zip(rec_texts, rec_scores):
                    if text:
                        lines.append({
                            "text": text,
                            "confidence": round(float(score), 4),
                            "bbox": [],
                        })
                        all_text.append(text)
            elif isinstance(r0, list):
                # PaddleOCR 2.x: [[box, (text, conf)], ...]
                for line in r0:
                    try:
                        box, (text, confidence) = line
                        flat_box = [round(v) for point in box for v in point]
                        lines.append({
                            "text": text,
                            "confidence": round(float(confidence), 4),
                            "bbox": flat_box,
                        })
                        all_text.append(text)
                    except (ValueError, TypeError):
                        pass

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
                          dpi: int = 200, lang: str = "ch",
                          cpu_threads: int = 0,
                          use_angle_cls: bool = True,
                          use_gpu: bool = False,
                          page_delay_ms: int = 0) -> dict:
    """
    对 PDF 指定页面执行 OCR。
    先逐页渲染为 PNG，再逐页识别。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"success": False, "error": "PyMuPDF (fitz) 未安装"}

    # CPU 线程限制 (必须在加载 BLAS 库之前设置)
    _limit_cpu_threads(cpu_threads)

    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"PDF 不存在: {pdf_path}"}

    # 抑制 Paddle 初始化日志（破坏 JSON stdout）
    with contextlib.redirect_stdout(io.StringIO()):
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=use_angle_cls,
            device="gpu" if use_gpu else "cpu",
        )

    t0 = time.time()
    doc = fitz.open(pdf_path)
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

            # 页面间冷却延迟 (防止 CPU 持续满载导致过热)
            if page_delay_ms > 0:
                time.sleep(page_delay_ms / 1000.0)

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
    """输出 JSON 到原始 stdout (绕过被重定向到 stderr 的 sys.stdout)"""
    try:
        fd = _orig_stdout_fd  # CLI 入口保存的原始 stdout
    except NameError:
        fd = sys.stdout
    json.dump(obj, fd, ensure_ascii=False)
    fd.write("\n")
    fd.flush()


def _run_ocr_batch(image_paths: List[str], lang: str = "ch",
                   cpu_threads: int = 0,
                   use_angle_cls: bool = True,
                   use_gpu: bool = False,
                   page_delay_ms: int = 0) -> List[dict]:
    """
    批量 OCR 多张图片 — 一次性加载模型，处理所有图片。

    用于子进程批量模式: 避免每张图片都重新加载 PaddleOCR 模型。
    """
    # CPU 线程限制
    _limit_cpu_threads(cpu_threads)

    # 先导入 torch (固定 DLL)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    # 抑制 Paddle 初始化日志（破坏 JSON stdout）
    with contextlib.redirect_stdout(io.StringIO()):
        from paddleocr import PaddleOCR

    t0 = time.time()
    # 抑制 Paddle 初始化日志
    with contextlib.redirect_stdout(io.StringIO()):
        ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=use_angle_cls,
            device="gpu" if use_gpu else "cpu",
        )

    results = []
    for i, img_path in enumerate(image_paths):
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
            if raw and len(raw) > 0:
                r0 = raw[0]
                # PaddleOCR 3.x: OCRResult dict
                if isinstance(r0, dict) and "rec_texts" in r0:
                    rec_texts = r0.get("rec_texts", [])
                    rec_scores = r0.get("rec_scores", []) or [0.0] * len(rec_texts)
                    for text, score in zip(rec_texts, rec_scores):
                        if text:
                            lines.append({
                                "text": text,
                                "confidence": round(float(score), 4),
                                "bbox": [],
                            })
                            all_text.append(text)
                elif isinstance(r0, list):
                    # PaddleOCR 2.x
                    for line in r0:
                        try:
                            box, (text, confidence) = line
                            flat_box = [round(v) for point in box for v in point]
                            lines.append({
                                "text": text,
                                "confidence": round(float(confidence), 4),
                                "bbox": flat_box,
                            })
                            all_text.append(text)
                        except (ValueError, TypeError):
                            pass

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

        # 页面间冷却延迟 (防止 CPU 持续满载导致过热)
        if page_delay_ms > 0 and i < len(image_paths) - 1:
            time.sleep(page_delay_ms / 1000.0)

    return results


# ===== CLI 入口 (子进程模式) =====
if __name__ == "__main__":
    # 保存原始文件描述符，后续 JSON 输出必须绕过被污染的 sys.stdout
    _orig_stdout_fd = os.fdopen(os.dup(1), "w", encoding="utf-8")
    # 将 stdout 重定向到 stderr — 模型加载日志走 stderr，JSON 走 _orig_stdout_fd
    sys.stdout = sys.stderr

    if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    if len(sys.argv) < 2:
        result = {"success": False, "error": "Usage: python ocr_engine.py <image_or_pdf_path> [--pages 1,2,3] [--batch] [--cpu-threads N] [--no-angle-cls] [--use-gpu] [--delay-ms N]"}
        _stdout_json(result)
        sys.exit(1)

    # 解析可选参数
    cpu_threads = 0
    use_angle_cls = True
    use_gpu = False
    page_delay_ms = 0

    if "--cpu-threads" in sys.argv:
        idx = sys.argv.index("--cpu-threads")
        if idx + 1 < len(sys.argv):
            cpu_threads = int(sys.argv[idx + 1])

    if "--no-angle-cls" in sys.argv:
        use_angle_cls = False

    if "--use-gpu" in sys.argv:
        use_gpu = True

    if "--delay-ms" in sys.argv:
        idx = sys.argv.index("--delay-ms")
        if idx + 1 < len(sys.argv):
            page_delay_ms = int(sys.argv[idx + 1])

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

        results = _run_ocr_batch(
            image_paths,
            cpu_threads=cpu_threads,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            page_delay_ms=page_delay_ms,
        )
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
        # PDF 多页 OCR (单进程内完成)
        result = _run_ocr_on_pdf_pages(
            input_path, pages,
            cpu_threads=cpu_threads,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            page_delay_ms=page_delay_ms,
        )
    elif input_path.lower().endswith('.pdf') and not pages:
        result = {"success": False, "error": "PDF 需要指定 --pages 参数"}
    else:
        # 单张图片 OCR
        result = _run_ocr_on_image(
            input_path,
            cpu_threads=cpu_threads,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
        )

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
                 use_subprocess: bool = True,
                 cpu_threads: int = 0,
                 use_angle_cls: bool = True,
                 use_gpu: bool = False,
                 page_delay_ms: int = 0,
                 ocr_tmp_dir: str = None):
        """
        Args:
            lang: OCR 语言 ("ch" = 中文)
            dpi: 渲染 PDF 页面的 DPI (越大越清晰，越慢)
            use_subprocess: 是否使用子进程隔离 (默认 True)
            cpu_threads: PaddlePaddle 推理线程数 (0=自动, 建议 2-4)
            use_angle_cls: 文本方向分类 (关闭可大幅降低 CPU)
            use_gpu: 使用 GPU 推理 (需 CUDA paddlepaddle-gpu)
            page_delay_ms: 页面间冷却延迟 (ms)，防止 CPU 持续满载过热
            ocr_tmp_dir: OCR 临时图片目录 (默认 E:/RongNengRAG/data/ocr_tmp)
                         必须不在系统盘，避免 C 盘爆满
        """
        self.lang = lang
        self.dpi = dpi
        self.use_subprocess = use_subprocess
        self.cpu_threads = cpu_threads
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.page_delay_ms = page_delay_ms
        self.ocr_tmp_dir = ocr_tmp_dir or r"E:\RongNengRAG\data\ocr_tmp"
        self._ocr_instance = None  # 直接模式下的缓存
        os.makedirs(self.ocr_tmp_dir, exist_ok=True)

    def _get_ocr(self):
        """获取 OCR 实例（直接模式，需 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python）"""
        if self._ocr_instance is None:
            _limit_cpu_threads(self.cpu_threads)
            from paddleocr import PaddleOCR
            self._ocr_instance = PaddleOCR(
                **_build_ocr_kwargs(self.lang, self.use_angle_cls,
                                   self.use_gpu, self.cpu_threads),
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

        # GPU 模式背压
        if self.use_gpu:
            _wait_for_gpu_slot("OCR (直接模式)")

        # 直接模式
        if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        _limit_cpu_threads(self.cpu_threads)
        return _run_ocr_on_image(
            image_path, self.lang,
            cpu_threads=self.cpu_threads,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
        )

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
            tmp_dir: 临时图片目录 (默认 self.ocr_tmp_dir, E盘)

        Returns:
            {"success": bool, "text": str, "pages": {...}, "elapsed_ms": float}
        """
        if not pages:
            return {"success": True, "text": "", "pages": {}, "total_pages_ocr": 0, "elapsed_ms": 0}

        if self.use_subprocess:
            return self._ocr_pdf_subprocess(pdf_path, pages,
                                            tmp_dir or self.ocr_tmp_dir)

        # GPU 模式背压
        if self.use_gpu:
            _wait_for_gpu_slot(f"OCR PDF ({len(pages)}页, 直接模式)")

        # 直接模式
        if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        _limit_cpu_threads(self.cpu_threads)
        return _run_ocr_on_pdf_pages(
            pdf_path, pages, self.dpi, self.lang,
            cpu_threads=self.cpu_threads,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
            page_delay_ms=self.page_delay_ms,
        )

    def _get_ocr_python(self) -> str:
        """OCR子进程 Python 路径 — GPU模式用PPOCRLabel venv (paddlepaddle-gpu 3.2.2)"""
        _OCR_VENV_PYTHON = r"E:\RongNengRAG\tools\PPOCRLabel\.venv\Scripts\python.exe"
        if self.use_gpu and os.path.exists(_OCR_VENV_PYTHON):
            return _OCR_VENV_PYTHON
        return sys.executable

    def _get_ocr_env(self):
        """
        返回 OCR 子进程的环境变量 — 隔离 conda base 路径，确保使用 venv 的包
        """
        env = os.environ.copy()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["PYTHONIOENCODING"] = "utf-8"
        env["GLOG_minloglevel"] = "2"
        # 清除 PYTHONPATH 防止 conda base 干扰 venv 包加载
        env.pop("PYTHONPATH", None)
        if self.cpu_threads > 0:
            threads_str = str(self.cpu_threads)
            for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                env[var] = threads_str
        return env

    def _ocr_subprocess(self, image_path: str) -> dict:
        """通过子进程调用 OCR (单图)"""
        import subprocess
        import tempfile

        script = Path(__file__).resolve()
        env = self._get_ocr_env()

        # GPU 模式: 启动子进程前等待显存释放 (LLM 优先)
        if self.use_gpu:
            _wait_for_gpu_slot("OCR (单页)")

        # 构建 CLI 参数
        py_exe = self._get_ocr_python()
        cmd = [py_exe, str(script), image_path]
        if self.cpu_threads > 0:
            cmd.extend(["--cpu-threads", str(self.cpu_threads)])
        if not self.use_angle_cls:
            cmd.append("--no-angle-cls")
        if self.use_gpu:
            cmd.append("--use-gpu")

        try:
            proc = subprocess.run(
                cmd,
                input=None, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,  # stderr 丢弃防死锁
                timeout=120, env=env,
                cwd=str(script.parent.parent.parent),
            )

            if proc.returncode != 0:
                return {"success": False, "error": f"OCR 子进程退出码 {proc.returncode}"}

            return _parse_json_stdout(proc.stdout) or {}

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
        env = self._get_ocr_env()

        # GPU 模式: 启动子进程前等待显存释放 (LLM 优先)
        if self.use_gpu:
            _wait_for_gpu_slot(f"OCR (PDF批量, {len(pages)}页)")

        # 构建批量 OCR 的 CLI 参数
        py_exe = self._get_ocr_python()
        batch_cmd = [py_exe, str(script), "--batch"]
        if self.cpu_threads > 0:
            batch_cmd.extend(["--cpu-threads", str(self.cpu_threads)])
        if not self.use_angle_cls:
            batch_cmd.append("--no-angle-cls")
        if self.use_gpu:
            batch_cmd.append("--use-gpu")
        if self.page_delay_ms > 0:
            batch_cmd.extend(["--delay-ms", str(self.page_delay_ms)])

        # 构建单页 OCR 回退的 CLI 参数
        single_cmd_base = [py_exe, str(script)]
        if self.cpu_threads > 0:
            single_cmd_base.extend(["--cpu-threads", str(self.cpu_threads)])
        if not self.use_angle_cls:
            single_cmd_base.append("--no-angle-cls")
        if self.use_gpu:
            single_cmd_base.append("--use-gpu")

        # 单次子进程处理全部页面 — 避免多次 subprocess.run 导致
        # WDDM 累积多个 CUDA context 撑爆显存
        BATCH_CHUNK_SIZE = 500  # 足够覆盖任何PDF(单次子进程)
        GPU_SEC_PER_PAGE = 20   # 保守超时预算

        t0 = time.time()
        doc = fitz.open(pdf_path)
        all_image_paths = []
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
                all_image_paths.append((page_num, img_path))

            # Step 2: 分批 OCR (每批 ≤ BATCH_CHUNK_SIZE 页)
            for chunk_start in range(0, len(all_image_paths), BATCH_CHUNK_SIZE):
                chunk = all_image_paths[chunk_start:chunk_start + BATCH_CHUNK_SIZE]
                chunk_pages = len(chunk)
                chunk_timeout = max(300, chunk_pages * GPU_SEC_PER_PAGE)

                print(f"   [OCR] 批次 {chunk_start // BATCH_CHUNK_SIZE + 1}/"
                      f"{(len(all_image_paths) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE}: "
                      f"{chunk_pages} 页, 超时 {chunk_timeout}s")

                img_list_json = json.dumps(
                    [p for _, p in chunk], ensure_ascii=False
                )

                batch_results = None
                try:
                    proc = subprocess.run(
                        batch_cmd,
                        input=img_list_json, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        timeout=chunk_timeout,
                        env=env,
                        cwd=str(script.parent.parent.parent),
                    )

                    if proc.returncode == 0 and proc.stdout.strip():
                        batch_results = _parse_json_stdout(proc.stdout) or []
                    else:
                        print(f"   [OCR] 批量 OCR 失败 (RC={proc.returncode})，回退逐页")
                except subprocess.TimeoutExpired:
                    print(f"   [OCR] 批次超时 ({chunk_timeout}s)，该批次回退逐页")
                except json.JSONDecodeError as e:
                    print(f"   [OCR] JSON 解析失败: {e}")

                # Step 3: 解析该批次结果
                if batch_results and isinstance(batch_results, list):
                    for i, (page_num, _) in enumerate(chunk):
                        if i < len(batch_results):
                            page_result = batch_results[i]
                            if page_result.get("success"):
                                pages_result[str(page_num + 1)] = {
                                    "lines": page_result.get("lines", []),
                                    "text": page_result.get("text", ""),
                                }
                                all_text_parts.append(page_result.get("text", ""))
                        total_pages += 1
                else:
                    # 回退：逐页 OCR
                    for page_num, img_path in chunk:
                        try:
                            proc = subprocess.run(
                                single_cmd_base + [img_path],
                                text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=180, env=env,
                                cwd=str(script.parent.parent.parent),
                            )
                            if proc.returncode == 0 and proc.stdout.strip():
                                page_result = _parse_json_stdout(proc.stdout) or {}
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
            for _, img_path in all_image_paths:
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
