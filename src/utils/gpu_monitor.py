"""
GPU 显存监控器 — 入库背压控制，优先保障 LLM 推理

功能:
  1. 实时检测 GPU 显存使用情况 (pynvml)
  2. 检测 Ollama 是否正在执行推理
  3. 入库前等待显存释放 (LLM 优先策略)
  4. 可配置的阈值和轮询间隔
"""

import time
import logging
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class GpuMonitor:
    """
    GPU 显存监控器

    用法:
        monitor = GpuMonitor()
        if monitor.wait_for_vram(min_free_mb=3000):
            # 显存充足，可以入库
            ...
    """

    def __init__(self,
                 min_free_vram_mb: int = 3000,
                 ollama_base_url: str = "http://localhost:11434",
                 poll_interval_s: float = 5.0,
                 max_wait_s: float = 600.0):
        """
        Args:
            min_free_vram_mb: 入库所需的最小空闲显存 (MB)
                             默认为 3000MB — 足够同时加载 PaddleOCR + 缓冲
            ollama_base_url: Ollama 服务地址
            poll_interval_s: 轮询间隔 (秒)
            max_wait_s: 最大等待时间 (秒)，超时后强制执行
        """
        self.min_free_vram_mb = min_free_vram_mb
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s
        self._nvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_available = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU 监控就绪: {self._device_count} 个设备 (pynvml)")
        except Exception as e:
            logger.warning(f"pynvml 不可用，回退到 torch.cuda: {e}")
            self._init_torch_fallback()

    def _init_torch_fallback(self):
        """回退到 torch.cuda 监控 (仅报告 PyTorch 显存，无法监控 Ollama)"""
        try:
            import torch
            self._torch = torch
            self._torch_available = torch.cuda.is_available()
            if self._torch_available:
                logger.info("GPU 监控就绪: torch.cuda (回退模式，仅限 PyTorch)")
            else:
                logger.warning("GPU 监控不可用: 无 CUDA 设备")
        except ImportError:
            self._torch_available = False
            logger.warning("GPU 监控不可用: torch 未安装")

    def get_vram_info(self) -> dict:
        """
        获取 GPU 显存信息

        Returns:
            {total_mb, used_mb, free_mb, device_index}
        """
        if self._nvml_available:
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
                info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "total_mb": info.total // (1024 * 1024),
                    "used_mb": info.used // (1024 * 1024),
                    "free_mb": info.free // (1024 * 1024),
                    "device_index": 0,
                }
            except Exception:
                pass

        if getattr(self, "_torch_available", False):
            try:
                total = self._torch.cuda.get_device_properties(0).total_memory
                reserved = self._torch.cuda.memory_reserved(0)
                allocated = self._torch.cuda.memory_allocated(0)
                total_mb = total // (1024 * 1024)
                used_mb = reserved // (1024 * 1024)
                free_mb = (total - reserved) // (1024 * 1024)
                return {
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "free_mb": free_mb,
                    "device_index": 0,
                    "note": "torch.cuda only — 不含其他进程",
                }
            except Exception:
                pass

        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "device_index": -1}

    def get_free_vram_mb(self) -> int:
        """返回当前空闲显存 (MB)"""
        return self.get_vram_info()["free_mb"]

    def is_ollama_busy(self, timeout_s: float = 3.0) -> bool:
        """
        检测 Ollama 是否正在执行推理

        通过 /api/ps 端点获取当前运行的模型列表，
        如果有模型正在运行，说明 LLM 正在处理请求。
        """
        try:
            resp = requests.get(
                f"{self.ollama_base_url}/api/ps",
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return len(models) > 0
        except Exception:
            pass
        return False

    def wait_for_vram(self,
                      min_free_mb: Optional[int] = None,
                      timeout_s: Optional[float] = None) -> bool:
        """
        等待 GPU 显存释放到指定阈值

        策略:
          1. 检查空闲显存 >= min_free_mb → 立即返回 True
          2. 空闲不足 → 检查 Ollama 是否在推理
          3. Ollama 繁忙 → 等待并轮询 (LLM 优先)
          4. Ollama 空闲但显存不足 → 可能是其他进程占用，等待较短时间后强制执行

        Args:
            min_free_mb: 所需最小空闲显存，默认使用实例配置
            timeout_s: 最大等待时间，默认使用实例配置

        Returns:
            True: 显存充足或等待后恢复
            False: 超时仍未恢复 (会强制执行)
        """
        if min_free_mb is None:
            min_free_mb = self.min_free_vram_mb
        if timeout_s is None:
            timeout_s = self.max_wait_s

        t_start = time.time()
        logged_busy = False

        while True:
            vram = self.get_vram_info()
            free_mb = vram["free_mb"]
            elapsed = time.time() - t_start

            if free_mb >= min_free_mb:
                if logged_busy:
                    logger.info(
                        f"显存恢复: {free_mb}MB 空闲 (等待了 {elapsed:.0f}s)"
                    )
                return True

            if elapsed >= timeout_s:
                logger.warning(
                    f"等待显存超时 ({elapsed:.0f}s)，当前空闲: {free_mb}MB，"
                    f"阈值: {min_free_mb}MB，强制执行"
                )
                return True  # 超时也继续，避免永久阻塞

            # 检查 Ollama 状态
            if not logged_busy:
                ollama_busy = self.is_ollama_busy()
                if ollama_busy:
                    logger.info(
                        f"显存不足 ({free_mb}MB < {min_free_mb}MB)，"
                        f"Ollama 正在推理，等待 LLM 释放显存..."
                    )
                    logged_busy = True
                else:
                    # Ollama 空闲但显存不足 — 可能是 BGE-M3 或其他进程占用
                    # 等待一段较短时间后强制执行
                    short_timeout = min(timeout_s, 30.0)
                    if elapsed >= short_timeout:
                        logger.warning(
                            f"显存持续不足 ({free_mb}MB < {min_free_mb}MB)，"
                            f"但 Ollama 空闲，强制执行入库"
                        )
                        return True

            time.sleep(self.poll_interval_s)

    def __del__(self):
        if self._nvml_available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


# 全局单例
_monitor: Optional[GpuMonitor] = None


def get_gpu_monitor(min_free_vram_mb: int = 3000) -> GpuMonitor:
    """获取全局 GPU 监控器单例"""
    global _monitor
    if _monitor is None:
        _monitor = GpuMonitor(min_free_vram_mb=min_free_vram_mb)
    return _monitor
