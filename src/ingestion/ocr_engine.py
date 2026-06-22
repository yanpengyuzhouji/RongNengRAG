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
import gc
import threading
import subprocess
import contextlib
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ===== 模块级 OCR 并发控制 =====
_ocr_lock = threading.Lock()           # 全局OCR锁: 同一时刻最多一个OCR池在运行
OCR_VRAM_PER_WORKER_MB = 3000          # 单个OCR子进程预估VRAM占用 (含PaddleOCR模型+buffer)
OCR_MAX_WORKERS = 2                    # 硬上限: Windows WDDM下>2个PaddleOCR实例GPU争抢反而更慢
OCR_DEFAULT_PAGES_PER_WORKER = 25      # 默认每worker处理页数 (影响并行粒度)


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

    import uuid
    tmpdir = os.path.join(self.ocr_tmp_dir, f"ocr_pages_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmpdir, exist_ok=True)

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
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
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


# ===== 鲁棒子进程 I/O =====

def _spawn_ocr_worker(image_paths: List[str],
                      ocr_python_exe: str,
                      script_path: str,
                      env: dict,
                      lang: str = "ch",
                      cpu_threads: int = 0,
                      use_angle_cls: bool = True,
                      use_gpu: bool = False,
                      page_delay_ms: int = 0,
                      timeout_s: int = 0,
                      cwd: str = None) -> List[dict]:
    """
    启动单个 OCR 子进程，鲁棒处理 I/O 和超时。

    与 subprocess.run() 的关键区别:
      - 使用 Popen + daemon线程异步消费 stderr → 杜绝管道死锁
      - stdout 流式读取，支持大 JSON (百页级)
      - 超时分阶段: terminate() → 3s → kill() → 确保无僵尸进程
      - 启动前在主进程侧做 gc + empty_cache

    Args:
        image_paths: PNG 图片路径列表
        ocr_python_exe: OCR 子进程使用的 Python 解释器路径
        script_path: ocr_engine.py 脚本路径
        env: 子进程环境变量
        lang, cpu_threads, use_angle_cls, use_gpu, page_delay_ms: OCR 参数
        timeout_s: 超时秒数 (0 = 自动: max(300, len * 20))
        cwd: 子进程工作目录

    Returns:
        OCR 结果列表 [{"success": bool, "text": str, "lines": [...], ...}, ...]
        失败时返回包含 error 字段的 dict 列表
    """
    # 主进程侧清理 — 将 PyTorch 缓存放回 CUDA，增加 WDDM 可回收池
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # 构建 CLI 命令
    batch_cmd = [ocr_python_exe, script_path, "--batch"]
    if cpu_threads > 0:
        batch_cmd.extend(["--cpu-threads", str(cpu_threads)])
    if not use_angle_cls:
        batch_cmd.append("--no-angle-cls")
    if use_gpu:
        batch_cmd.append("--use-gpu")
    if page_delay_ms > 0:
        batch_cmd.extend(["--delay-ms", str(page_delay_ms)])

    if timeout_s <= 0:
        # 模型加载 ~30s + 每页 ~30s (GPU保守估计)
        timeout_s = 60 + len(image_paths) * 30

    img_list_json = json.dumps(image_paths, ensure_ascii=False)
    chunk_tag = f"OCR-{len(image_paths)}p"
    results = []

    try:
        proc = subprocess.Popen(
            batch_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd,
        )

        try:
            # communicate() 正确消费 stdout + stderr — 杜绝管道死锁
            # 内部用线程并发读取 stdout/stderr，不会阻塞子进程写入
            stdout_text, stderr_text = proc.communicate(
                input=img_list_json,
                timeout=timeout_s,
            )

            if stderr_text:
                # 只记录错误级别的 stderr 输出
                for line in stderr_text.split("\n"):
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in ("error", "fatal", "oom", "traceback", "exception")):
                        logger.warning(f"[{chunk_tag}] {line.rstrip()}")

            if proc.returncode != 0:
                logger.warning(f"[{chunk_tag}] 子进程退出码 {proc.returncode}")
                return [{"success": False, "error": f"OCR 子进程退出码 {proc.returncode}"}
                        for _ in image_paths]

        except subprocess.TimeoutExpired:
            logger.warning(f"[{chunk_tag}] 超时 ({timeout_s}s)，开始终止进程树...")
            _kill_process_tree(proc.pid)
            return [{"success": False, "error": f"OCR 超时 ({timeout_s}s)"}
                    for _ in image_paths]

        except Exception as e:
            logger.error(f"[{chunk_tag}] 子进程异常: {e}")
            _kill_process_tree(proc.pid)
            return [{"success": False, "error": str(e)} for _ in image_paths]

        finally:
            # 确保进程树已清理
            if proc.poll() is None:
                _kill_process_tree(proc.pid)

        # 解析 stdout JSON
        if stdout_text and stdout_text.strip():
            parsed = _parse_json_stdout(stdout_text)
            if parsed and isinstance(parsed, list):
                # 成功: 补齐缺失的结果
                for i in range(len(image_paths)):
                    if i < len(parsed):
                        results.append(parsed[i])
                    else:
                        results.append({
                            "success": False,
                            "error": "OCR 结果缺失",
                            "lines": [],
                            "text": "",
                        })
                return results
            elif parsed and isinstance(parsed, dict):
                # 单结果包装成列表
                return [parsed]
            else:
                logger.warning(f"[{chunk_tag}] stdout 解析失败: "
                               f"{stdout_text[:200]}")
                return [{"success": False, "error": "OCR JSON 解析失败"}
                        for _ in image_paths]
        else:
            return [{"success": False, "error": "OCR 无输出"}
                    for _ in image_paths]

    except FileNotFoundError:
        logger.error(f"[{chunk_tag}] OCR Python 不存在: {ocr_python_exe}")
        return [{"success": False, "error": f"OCR Python 不存在: {ocr_python_exe}"}
                for _ in image_paths]
    except Exception as e:
        logger.error(f"[{chunk_tag}] 未预期异常: {e}")
        return [{"success": False, "error": str(e)} for _ in image_paths]


# ===== VRAM 感知 OCR 进程池 =====

def _auto_ocr_workers(use_gpu: bool) -> int:
    """
    基于当前 GPU 显存状况自动确定 OCR 并行数。

    Returns:
        ≥1 的安全并行数。use_gpu=False 时返回 min(2, cpu_count//2)。
    """
    if not use_gpu:
        cpu_count = os.cpu_count() or 4
        return max(1, min(OCR_MAX_WORKERS, cpu_count // 2))

    try:
        src_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
        sys.path.insert(0, os.path.dirname(os.path.dirname(src_dir)))
        from src.utils.gpu_monitor import get_gpu_monitor
        monitor = get_gpu_monitor()
        return monitor.get_ocr_workers_possible(
            vram_per_worker_mb=OCR_VRAM_PER_WORKER_MB,
            safety_margin_mb=1500,
        )
    except Exception:
        return 1


def ocr_pool_map(image_paths: List[str],
                 ocr_python_exe: str,
                 script_path: str,
                 env: dict,
                 lang: str = "ch",
                 cpu_threads: int = 0,
                 use_angle_cls: bool = True,
                 use_gpu: bool = False,
                 page_delay_ms: int = 0,
                 max_workers: int = 0,
                 pages_per_worker: int = OCR_DEFAULT_PAGES_PER_WORKER,
                 cwd: str = None,
                 on_progress=None) -> List[dict]:
    """
    VRAM 感知 OCR 进程池 — 并行处理图片列表。

    设计:
      - max_workers=0 (默认) → 基于 effective_free_vram 自动确定
      - max_workers=1 → 单进程直通 (安全模式)
      - max_workers≥2 → ThreadPool 同时启动 N 个 OCR 子进程
      - 使用 ThreadPoolExecutor (非 ProcessPoolExecutor): 每个线程 spawn
        一个独立的 Popen 子进程，避免 pickle 序列化开销

    Args:
        image_paths: 待OCR的PNG图片路径列表
        ocr_python_exe: OCR 子进程 Python 解释器
        script_path: ocr_engine.py 绝对路径
        env: 子进程环境变量字典
        lang, cpu_threads, use_angle_cls, use_gpu, page_delay_ms: OCR参数
        max_workers: 0=自动, 1=单进程, 2-3=手动
        pages_per_worker: 每个worker处理的最大页数 (影响分块粒度)
        cwd: 子进程工作目录
        on_progress: 进度回调 (done: int, total: int)

    Returns:
        OCR 结果列表，与 image_paths 顺序对应
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(image_paths)
    if total == 0:
        return []

    # 确定并发数
    if max_workers <= 0:
        max_workers = _auto_ocr_workers(use_gpu)

    max_workers = max(1, min(OCR_MAX_WORKERS, max_workers))
    if max_workers == 1 or total <= pages_per_worker:
        # 单进程直通 — 无进程池开销
        if on_progress:
            on_progress(0, total)
        results = _spawn_ocr_worker(
            image_paths,
            ocr_python_exe=ocr_python_exe,
            script_path=script_path,
            env=env,
            lang=lang,
            cpu_threads=cpu_threads,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            page_delay_ms=page_delay_ms,
            cwd=cwd,
        )
        if on_progress:
            on_progress(total, total)
        return results

    # 多进程池模式 — 将图片分块给多个 worker
    chunk_size = max(pages_per_worker, total // max_workers)
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append(image_paths[i:i + chunk_size])

    # 如果分块后只有1块，单进程处理
    if len(chunks) <= 1:
        if on_progress:
            on_progress(0, total)
        results = _spawn_ocr_worker(
            image_paths,
            ocr_python_exe=ocr_python_exe,
            script_path=script_path,
            env=env,
            lang=lang,
            cpu_threads=cpu_threads,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            page_delay_ms=page_delay_ms,
            cwd=cwd,
        )
        if on_progress:
            on_progress(total, total)
        return results

    logger.info(f"[OCR Pool] {total} 页 → {len(chunks)} 批 × ≤{max_workers} workers "
                f"(每批≤{chunk_size}页, VRAM≈{max_workers * OCR_VRAM_PER_WORKER_MB // 1000}GB)")

    # ThreadPool 并行启动子进程
    ordered_results = [None] * len(chunks)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, chunk in enumerate(chunks):
            future = pool.submit(
                _spawn_ocr_worker,
                chunk,
                ocr_python_exe=ocr_python_exe,
                script_path=script_path,
                env=env,
                lang=lang,
                cpu_threads=cpu_threads,
                use_angle_cls=use_angle_cls,
                use_gpu=use_gpu,
                page_delay_ms=page_delay_ms,
                cwd=cwd,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                chunk_result = future.result()
                ordered_results[idx] = chunk_result
                completed_count += len(chunks[idx])
                if on_progress:
                    on_progress(completed_count, total)
            except Exception as e:
                logger.error(f"[OCR Pool] Worker {idx} 异常: {e}")
                ordered_results[idx] = [
                    {"success": False, "error": str(e)}
                    for _ in chunks[idx]
                ]
                completed_count += len(chunks[idx])
                if on_progress:
                    on_progress(completed_count, total)

    # 按原始顺序合并结果
    merged = []
    for chunk_result in ordered_results:
        if chunk_result:
            merged.extend(chunk_result)
        else:
            # 异常丢失 — 补占位
            merged.extend([{"success": False, "error": "Worker 结果丢失"}])

    # 确保长度匹配
    while len(merged) < total:
        merged.append({"success": False, "error": "结果缺失"})
    return merged[:total]


# ===== CLI 入口 (子进程模式) =====
if __name__ == "__main__":
    # 保存原始文件描述符，后续 JSON 输出必须绕过被污染的 sys.stdout
    _orig_stdout_fd = os.fdopen(os.dup(1), "w", encoding="utf-8")
    # 将 stdout 重定向到 stderr — 模型加载日志走 stderr，JSON 走 _orig_stdout_fd
    sys.stdout = sys.stderr

    if not os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    if len(sys.argv) < 2:
        result = {"success": False, "error": "Usage: python ocr_engine.py <image_or_pdf_path> [--pages 1,2,3] [--batch] [--stream] [--cpu-threads N] [--no-angle-cls] [--use-gpu] [--delay-ms N]"}
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

    # --stream 模式: 长连接流式OCR
    #   stdin:  一行一个图片路径 (UTF-8, newline分隔)
    #   stdout: 一行一个 JSON 结果 (_orig_stdout_fd)
    #   EOF 时退出
    if "--stream" in sys.argv:
        _limit_cpu_threads(cpu_threads)
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

        with contextlib.redirect_stdout(io.StringIO()):
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(
                lang="ch",
                use_textline_orientation=use_angle_cls,
                device="gpu" if use_gpu else "cpu",
            )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            img_path = line

            if not os.path.exists(img_path):
                _stdout_json({"success": False, "error": f"文件不存在: {img_path}", "lines": [], "text": ""})
                continue

            try:
                raw = ocr.ocr(img_path)
                lines_out = []
                all_text = []
                if raw and len(raw) > 0:
                    r0 = raw[0]
                    if isinstance(r0, dict) and "rec_texts" in r0:
                        rec_texts = r0.get("rec_texts", [])
                        rec_scores = r0.get("rec_scores", []) or [0.0] * len(rec_texts)
                        for text, score in zip(rec_texts, rec_scores):
                            if text:
                                lines_out.append({"text": text, "confidence": round(float(score), 4), "bbox": []})
                                all_text.append(text)
                    elif isinstance(r0, list):
                        for item in r0:
                            try:
                                box, (text, confidence) = item
                                lines_out.append({"text": text, "confidence": round(float(confidence), 4), "bbox": []})
                                all_text.append(text)
                            except (ValueError, TypeError):
                                pass

                _stdout_json({"success": True, "text": "\n".join(all_text), "lines": lines_out})
            except Exception as e:
                _stdout_json({"success": False, "error": str(e), "lines": [], "text": ""})

        sys.exit(0)

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


# ===== 流式OCR辅助函数 =====

# pymupdf/mupdf 的 get_pixmap() 不是线程安全的 — 并发渲染会导致 C 层
# fz_run_display_list 访问冲突 (exit code 139 SIGSEGV)。所有渲染调用必须串行。
_pymupdf_render_lock = threading.Lock()


def _render_page_to_png(doc, page_num: int, dpi: int, tmp_dir: str,
                        max_dim: int = 0) -> Optional[str]:
    """渲染单页PDF为PNG (内部有 pymupdf 全局锁，安全但串行)

    Args:
        max_dim: 图片最大边长 (px)。超过此值时用 LANCZOS 缩放。
                 0 = 不缩放。默认 3000 可避免超大工程图打满 GPU 显存。
                 示例: 9744×6890 → 3000×2121, OCR 从 660s 降至 2.6s。
    """
    # 在锁内完成 pymupdf 渲染，锁外做 Pillow 缩放和文件 I/O
    with _pymupdf_render_lock:
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(tmp_dir, f"page_{page_num + 1}.png")

        need_resize = max_dim > 0 and max(pix.width, pix.height) > max_dim
        if need_resize:
            # 在锁内提取 pix 原始数据，然后释放 mupdf 对象
            samples = pix.samples
            width, height = pix.width, pix.height
            del pix, page
        else:
            pix.save(img_path)
            return img_path

    # 锁外: Pillow 缩放 + 写盘 (纯 Python/PIL, 不碰 mupdf)
    try:
        from PIL import Image
        img = Image.frombytes("RGB", [width, height], samples)
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        img.save(img_path)
        return img_path
    except ImportError:
        pass
    # Pillow 不可用，回退: 重新渲染 (极少数情况)
    with _pymupdf_render_lock:
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(img_path)
        return img_path


def _kill_process_tree(pid: int):
    """Windows: 杀整个进程树（子进程+孙进程）

    subprocess.Popen.kill() 只杀直接子进程，PaddlePaddle 的 CUDA 后端
    作为孙进程存活 → 显存泄漏。taskkill /T 可级联杀树。

    注意: bash 下 /F 会被转换为路径 F:/，必须用 cmd //c 包装。
    """
    try:
        subprocess.run(
            ["cmd", "//c", f"taskkill /F /T /PID {pid}"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # 进程已死或权限不足


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
                 ocr_tmp_dir: str = None,
                 turbo: bool = False,
                 turbo_max_workers: int = 0,
                 max_image_dim: int = 3000):
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
            turbo: 启用 OCR 进程池 (多子进程并行, VRAM感知)
            turbo_max_workers: turbo模式最大并行数 (0=自动, 1=单进程safe, 2-3=手动)
            max_image_dim: 图片最大边长 (px)。大规格工程图纸渲染后可能达到
                          9744×6890，直接送入 PaddleOCR 会导致单页 660s +
                          12GB 显存峰值。缩放至 3000px 后降至 2.6s + 10GB。
                          0 = 不缩放。
        """
        self.lang = lang
        self.dpi = dpi
        self.use_subprocess = use_subprocess
        self.cpu_threads = cpu_threads
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.page_delay_ms = page_delay_ms
        self.ocr_tmp_dir = ocr_tmp_dir or r"E:\RongNengRAG\data\ocr_tmp"
        self.max_image_dim = max_image_dim
        self.turbo = turbo
        self.turbo_max_workers = turbo_max_workers
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
        with _ocr_lock:
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
        2. 调用 OCR 子进程识别每张图片 (全局锁保护)
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

        # 全局 OCR 锁 — 防止并发 OCR 池 (FastAPI 多请求场景)
        # turbo模式下尤其重要: 2个请求同时开2-worker池 = 4子进程 = VRAM爆炸
        with _ocr_lock:
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
        """通过子进程调用 OCR (单图) — 使用鲁棒 I/O"""
        script = Path(__file__).resolve()
        env = self._get_ocr_env()
        py_exe = self._get_ocr_python()
        cwd = str(script.parent.parent.parent)

        # GPU 模式: 启动子进程前等待显存释放 (LLM 优先)
        if self.use_gpu:
            _wait_for_gpu_slot("OCR (单页)")

        results = _spawn_ocr_worker(
            [image_path],
            ocr_python_exe=py_exe,
            script_path=str(script),
            env=env,
            lang=self.lang,
            cpu_threads=self.cpu_threads,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
            page_delay_ms=self.page_delay_ms,
            timeout_s=180,
            cwd=cwd,
        )
        return results[0] if results else {"success": False, "error": "OCR 无结果"}

    def _ocr_pdf_subprocess(self, pdf_path: str, pages: List[int],
                            tmp_dir: str = None) -> dict:
        """通过 _spawn_ocr_worker 处理 PDF 扫描页 — 渲染 → 批量OCR → 清理

        替代旧 --stream 长连接方案，使用 communicate() 模式的子进程:
          - communicate() 内部用线程并发读写 stdin/stdout/stderr → 零死锁风险
          - 超时后 kill 整个进程树 → 无孤儿进程泄漏
          - 单 Popen → 单 PaddleOCR 实例 → 无 GPU 争抢
        """
        import fitz
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if tmp_dir is None:
            import uuid
            tmp_dir = os.path.join(self.ocr_tmp_dir, f"ocr_batch_{uuid.uuid4().hex[:8]}")
            os.makedirs(tmp_dir, exist_ok=True)
            cleanup_dir = tmp_dir
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            cleanup_dir = None

        t0 = time.time()

        # GPU 背压 + Ollama 显存释放
        if self.use_gpu:
            self._free_gpu_for_ocr(len(pages))

        # ===== Step 1: 渲染全部页面为 PNG (CPU 密集, 线程池加速) =====
        doc = fitz.open(pdf_path)
        page_img_map = {}  # page_num (0-based) → img_path
        total = len(pages)

        try:
            max_render = min(8, max(4, total))
            render_pool = ThreadPoolExecutor(max_workers=max_render,
                                             thread_name_prefix="render")
            futures = {}
            for pg in pages:
                if pg < 0 or pg >= len(doc):
                    continue
                futures[render_pool.submit(
                    _render_page_to_png, doc, pg, self.dpi, tmp_dir,
                    self.max_image_dim,
                )] = pg

            render_ok = 0
            for future in as_completed(futures):
                pg = futures[future]
                try:
                    img_path = future.result()
                    if img_path:
                        page_img_map[pg] = img_path
                        render_ok += 1
                except Exception as e:
                    logger.warning(f"[OCR] 渲染p{pg+1}失败: {e}")
            render_pool.shutdown(wait=True)

            t_render = time.time() - t0
            print(f"   [OCR] 渲染完成: {render_ok}/{total} 页 ({t_render:.0f}s)")

            if not page_img_map:
                doc.close()
                return {"success": True, "text": "", "pages": {},
                        "total_pages_ocr": 0, "elapsed_ms": (time.time() - t0) * 1000}

        finally:
            doc.close()

        # ===== Step 2: 批量 OCR (单子进程, communicate 模式) =====
        img_list = list(page_img_map.values())
        pg_order = list(page_img_map.keys())  # 与 img_list 顺序对应

        script = Path(__file__).resolve()
        env = self._get_ocr_env()
        py_exe = self._get_ocr_python()
        cwd = str(script.parent.parent.parent)

        results = _spawn_ocr_worker(
            img_list,
            ocr_python_exe=py_exe,
            script_path=str(script),
            env=env,
            lang=self.lang,
            cpu_threads=self.cpu_threads,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
            page_delay_ms=self.page_delay_ms,
            cwd=cwd,
        )

        # ===== Step 3: 解析结果 + 清理 PNG =====
        pages_result = {}
        all_text_parts = []

        for i, result_obj in enumerate(results):
            pg = pg_order[i] if i < len(pg_order) else -1
            if result_obj.get("success"):
                pages_result[str(pg + 1)] = {
                    "lines": result_obj.get("lines", []),
                    "text": result_obj.get("text", ""),
                }
                text = result_obj.get("text", "")
                if text.strip():
                    all_text_parts.append(text)

            # 及时删除 PNG 释放磁盘
            if i < len(img_list):
                try:
                    os.remove(img_list[i])
                except OSError:
                    pass

        # 清理遗漏的 PNG
        for img_path in img_list:
            if os.path.isfile(img_path):
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        if cleanup_dir:
            try:
                import shutil
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            except Exception:
                pass

        total_pages = len(pages_result)
        elapsed = (time.time() - t0) * 1000
        valid_pages = sum(1 for v in pages_result.values()
                          if v.get("text", "").strip())

        if valid_pages > 0:
            print(f"   [OCR] {valid_pages}/{total} 页有效识别, "
                  f"耗时 {elapsed:.0f}ms ({elapsed/total_pages:.0f}ms/页)"
                  if total_pages else f"   [OCR] 耗时 {elapsed:.0f}ms")

        return {
            "success": True,
            "text": "\n".join(all_text_parts),
            "pages": pages_result,
            "total_pages_ocr": total_pages,
            "elapsed_ms": round(elapsed, 1),
        }

    def _free_gpu_for_ocr(self, page_count: int):
        """OCR 前释放 GPU 显存: 背压等待 + 通知 Ollama 释放模型"""
        _wait_for_gpu_slot(f"OCR ({page_count}页)")

        # 通知 Ollama 卸载模型 (下次请求时自动重新加载)
        try:
            import requests
            requests.post("http://localhost:11434/api/generate",
                          json={"model": "", "keep_alive": 0},
                          timeout=3)
        except Exception:
            pass  # Ollama 不在线或无影响

        try:
            from src.utils.gpu_monitor import log_vram_snapshot
            log_vram_snapshot(f"OCR前 ({page_count}页)")
        except Exception:
            pass
