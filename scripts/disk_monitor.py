"""
资源监控守护进程 — 同时监控磁盘 / 内存 / GPU 显存
- 磁盘 (C:): 剩余空间低于阈值时告警并杀进程
- 内存 (RAM): 使用率 >= 阈值时告警并杀进程
- GPU 显存: 使用率 >= 阈值时告警并杀进程
- 任一资源触发阈值 → 终止项目进程 → 监控退出

用法:
  python scripts/disk_monitor.py                              # 全部默认阈值
  python scripts/disk_monitor.py --disk-threshold 3           # 磁盘阈值 3GB
  python scripts/disk_monitor.py --max-ram 90 --max-gpu 90    # RAM/GPU 阈值 90%
  python scripts/disk_monitor.py --no-gpu                     # 跳过 GPU 监控
"""

import os
import sys
import time
import signal
import argparse
import logging
import subprocess
from dataclasses import dataclass

# ---- 配置日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Monitor] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("resource_monitor")

# ---- 要监控的项目进程特征 ----
PROCESS_MARKERS = [
    "src/api/main.py",
    "src/ui/app.py",
    "uvicorn",
    "gradio",
    "streamlit run",
]

# ---- 默认阈值 ----
DISK_THRESHOLD_GB_DEFAULT = 2.0        # C 盘剩余 < 2GB
MIN_RAM_FREE_GB_DEFAULT = 0.0          # 可用内存 < N GB (0=禁用)
MAX_RAM_PCT_DEFAULT = 95               # 内存使用率 >= 95%
MAX_GPU_PCT_DEFAULT = 95               # GPU 显存使用率 >= 95%
MAX_RUNTIME_MINUTES_DEFAULT = 0        # 最大运行时长 (0=不限时)
CHECK_INTERVAL_DEFAULT = 30            # 秒


# ═══════════════════════════════════════════════════════════════
# 资源采集
# ═══════════════════════════════════════════════════════════════

def get_disk_free_gb(drive: str = "C:") -> float:
    """获取指定盘符剩余空间 (GB)"""
    if sys.platform == "win32":
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive + "\\"),
            ctypes.byref(free_bytes),
            ctypes.byref(total_bytes),
            None,
        )
        return free_bytes.value / (1024 ** 3)
    else:
        stat = os.statvfs(drive)
        return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def get_ram_usage() -> tuple[float, float, float]:
    """
    返回 (已用_GB, 总量_GB, 使用率_%)
    使用 psutil (跨平台，不含缓存)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        used_gb = (mem.total - mem.available) / (1024 ** 3)  # 可用内存 = 真正可分配的内存
        pct = mem.percent
        return used_gb, total_gb, pct
    except ImportError:
        log.warning("psutil 未安装，跳过内存监控")
        return 0, 0, 0


def get_gpu_usage_pynvml() -> tuple[float, float, float] | None:
    """
    使用 pynvml 获取 GPU 显存 (第一块 NVIDIA GPU)
    返回 (已用_GB, 总量_GB, 使用率_%) 或 None
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / (1024 ** 3)
        used_gb = info.used / (1024 ** 3)
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
        pynvml.nvmlShutdown()
        return used_gb, total_gb, pct
    except Exception:
        return None


def get_gpu_usage_nvidia_smi() -> tuple[float, float, float] | None:
    """
    使用 nvidia-smi 命令获取 GPU 显存
    返回 (已用_MB→GB, 总量_MB→GB, 使用率_%) 或 None
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        if len(parts) < 2:
            return None
        used_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024
        pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
        return used_gb, total_gb, pct
    except Exception:
        return None


def get_gpu_usage() -> tuple[float, float, float] | None:
    """获取 GPU 显存使用情况 (优先 pynvml，回退 nvidia-smi)"""
    result = get_gpu_usage_pynvml()
    if result is not None:
        return result
    return get_gpu_usage_nvidia_smi()


@dataclass
class ResourceSnapshot:
    """一次采集的所有资源快照"""
    disk_free_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_pct: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_total_gb: float = 0.0
    gpu_pct: float = 0.0
    gpu_available: bool = True


def collect_resources(monitor_gpu: bool = True) -> ResourceSnapshot:
    """采集所有资源"""
    snap = ResourceSnapshot()

    snap.disk_free_gb = get_disk_free_gb("C:")
    snap.ram_used_gb, snap.ram_total_gb, snap.ram_pct = get_ram_usage()

    if monitor_gpu:
        gpu = get_gpu_usage()
        if gpu is not None:
            snap.gpu_used_gb, snap.gpu_total_gb, snap.gpu_pct = gpu
        else:
            snap.gpu_available = False

    return snap


# ═══════════════════════════════════════════════════════════════
# 进程管理
# ═══════════════════════════════════════════════════════════════

def find_project_pids() -> list[tuple[int, str]]:
    """找出属于本项目的进程 PID 和命令行"""
    pids: list[tuple[int, str]] = []

    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                line_lower = line.lower()
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                try:
                    pid = int(parts[1].strip())
                except ValueError:
                    continue

                cmd = ",".join(parts[2:]).strip()
                if not cmd:
                    continue

                # 排除自身
                if "disk_monitor" in cmd:
                    continue

                for marker in PROCESS_MARKERS:
                    if marker.lower() in cmd.lower():
                        # 对 python xxx.py 做更精确的匹配
                        if marker.endswith(".py"):
                            if f"python" in cmd.lower() and marker.split("/")[-1].lower() in cmd.lower():
                                pids.append((pid, cmd))
                                break
                        else:
                            if marker.lower() in cmd.lower():
                                pids.append((pid, cmd))
                                break

        except Exception as e:
            log.warning(f"wmic 查询失败: {e}")
    else:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "uvicorn|gradio|streamlit|main.py|app.py"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().splitlines():
                pid_str = pid_str.strip()
                if not pid_str:
                    continue
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == os.getpid():
                    continue
                try:
                    ps = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "command="],
                        capture_output=True, text=True, timeout=3,
                    )
                    cmd = ps.stdout.strip()
                except Exception:
                    cmd = ""
                if "disk_monitor" in cmd:
                    continue
                pids.append((pid, cmd))
        except Exception as e:
            log.warning(f"pgrep 查询失败: {e}")

    return pids


def kill_process(pid: int, cmd: str = "") -> bool:
    """发送 SIGTERM 终止进程"""
    try:
        os.kill(pid, signal.SIGTERM)
        log.info(f"  已发送 SIGTERM → PID {pid}: {cmd[:120]}")
        return True
    except PermissionError:
        log.error(f"  权限不足，无法终止 PID {pid}")
        return False
    except ProcessLookupError:
        log.info(f"  PID {pid} 已不存在")
        return True
    except Exception as e:
        log.error(f"  终止 PID {pid} 失败: {e}")
        return False


def force_kill(pid: int, cmd: str = "") -> bool:
    """强制结束进程 (taskkill /F 或 SIGKILL)"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        else:
            os.kill(pid, signal.SIGKILL)
        log.info(f"  已强制终止 → PID {pid}: {cmd[:120]}")
        return True
    except Exception as e:
        log.error(f"  强制终止 PID {pid} 失败: {e}")
        return False


def kill_all_project_processes():
    """两轮杀死所有项目进程 (SIGTERM → 等待3s → SIGKILL)"""
    pids = find_project_pids()
    if not pids:
        log.info("  未发现运行中的项目进程")
        return

    log.warning(f"  发现 {len(pids)} 个项目进程")

    # 第一轮: SIGTERM 优雅终止
    for pid, cmd in pids:
        kill_process(pid, cmd)

    # 等待 3 秒
    time.sleep(3)

    # 第二轮: 检查存活并强制终止
    remaining = find_project_pids()
    if remaining:
        log.warning(f"  仍有 {len(remaining)} 个进程存活，执行强制终止")
        for pid, cmd in remaining:
            force_kill(pid, cmd)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def _check_thresholds(
    snap: ResourceSnapshot,
    disk_threshold_gb: float,
    min_ram_free_gb: float,
    max_ram_pct: float,
    max_gpu_pct: float,
    monitor_gpu: bool,
) -> str | None:
    """
    检查所有资源是否触发阈值。
    返回触发原因字符串，未触发返回 None。
    """
    # 磁盘检查
    if snap.disk_free_gb < disk_threshold_gb:
        return (
            f"磁盘 C: 剩余 {snap.disk_free_gb:.2f} GB < 阈值 {disk_threshold_gb} GB"
        )

    # 内存检查 — 可用内存绝对阈值
    if min_ram_free_gb > 0 and snap.ram_total_gb > 0:
        ram_free = snap.ram_total_gb - snap.ram_used_gb
        if ram_free < min_ram_free_gb:
            return (
                f"可用内存 {ram_free:.2f} GB < 阈值 {min_ram_free_gb} GB "
                f"(已用 {snap.ram_pct:.1f}%, {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB)"
            )

    # 内存检查 — 使用率阈值
    if snap.ram_pct >= max_ram_pct:
        return (
            f"内存使用率 {snap.ram_pct:.1f}% >= 阈值 {max_ram_pct}% "
            f"({snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB)"
        )

    # GPU 显存检查
    if monitor_gpu and snap.gpu_available and snap.gpu_pct >= max_gpu_pct:
        return (
            f"GPU 显存使用率 {snap.gpu_pct:.1f}% >= 阈值 {max_gpu_pct}% "
            f"({snap.gpu_used_gb:.1f}/{snap.gpu_total_gb:.1f} GB)"
        )

    return None


def main():
    parser = argparse.ArgumentParser(
        description="资源监控守护进程 — 磁盘 / 内存 / GPU 显存"
    )
    parser.add_argument(
        "--disk-threshold", type=float, default=DISK_THRESHOLD_GB_DEFAULT,
        help=f"C盘剩余空间告警阈值 (GB)，默认 {DISK_THRESHOLD_GB_DEFAULT}",
    )
    parser.add_argument(
        "--min-ram-free", type=float, default=MIN_RAM_FREE_GB_DEFAULT,
        help=f"可用内存告警阈值 (GB)，默认 {MIN_RAM_FREE_GB_DEFAULT} (禁用)",
    )
    parser.add_argument(
        "--max-ram", type=float, default=MAX_RAM_PCT_DEFAULT,
        help=f"内存使用率告警阈值 (%%)，默认 {MAX_RAM_PCT_DEFAULT}",
    )
    parser.add_argument(
        "--max-gpu", type=float, default=MAX_GPU_PCT_DEFAULT,
        help=f"GPU 显存使用率告警阈值 (%%)，默认 {MAX_GPU_PCT_DEFAULT}",
    )
    parser.add_argument(
        "--max-runtime-minutes", type=float, default=MAX_RUNTIME_MINUTES_DEFAULT,
        help=f"最大运行时长 (分钟)，默认 {MAX_RUNTIME_MINUTES_DEFAULT} (不限时)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="跳过 GPU 监控",
    )
    parser.add_argument(
        "--interval", type=int, default=CHECK_INTERVAL_DEFAULT,
        help=f"检测间隔 (秒)，默认 {CHECK_INTERVAL_DEFAULT}",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="仅检测一次，不循环",
    )
    parser.add_argument(
        "--no-kill", action="store_true",
        help="仅告警不杀进程 (dry-run)",
    )
    args = parser.parse_args()

    monitor_gpu = not args.no_gpu

    log.info("=" * 60)
    log.info("  资源监控守护进程已启动")
    log.info(f"  磁盘告警阈值: C: 剩余 < {args.disk_threshold} GB")
    if args.min_ram_free > 0:
        log.info(f"  可用内存告警阈值: < {args.min_ram_free} GB")
    log.info(f"  内存使用率告警阈值: >= {args.max_ram}%")
    if monitor_gpu:
        log.info(f"  GPU 告警阈值: >= {args.max_gpu}%")
    else:
        log.info(f"  GPU 监控: 已禁用")
    if args.max_runtime_minutes > 0:
        log.info(f"  最大运行时长: {args.max_runtime_minutes} 分钟 (到时自动终止)")
    log.info(f"  检测间隔: {args.interval} 秒")
    log.info(f"  模式: {'仅告警 (dry-run)' if args.no_kill else '告警 + 杀进程'}")
    log.info("=" * 60)

    alert_count = 0
    start_time = time.time()
    max_runtime_s = args.max_runtime_minutes * 60 if args.max_runtime_minutes > 0 else 0

    try:
        while True:
            snap = collect_resources(monitor_gpu=monitor_gpu)

            # ---- 实时日志 ----
            parts = [f"磁盘剩余 {snap.disk_free_gb:.2f} GB"]
            if snap.ram_total_gb > 0:
                parts.append(f"RAM {snap.ram_pct:.1f}% ({snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB)")
            if monitor_gpu and snap.gpu_available:
                parts.append(f"GPU {snap.gpu_pct:.1f}% ({snap.gpu_used_gb:.1f}/{snap.gpu_total_gb:.1f} GB)")
            elif monitor_gpu and not snap.gpu_available:
                parts.append("GPU N/A")
            log.info(" | ".join(parts))

            # ---- 阈值检查 ----
            reason = _check_thresholds(
                snap,
                args.disk_threshold,
                args.min_ram_free,
                args.max_ram,
                args.max_gpu,
                monitor_gpu,
            )

            # ---- 运行时长检查 ----
            if max_runtime_s > 0 and not reason:
                elapsed = time.time() - start_time
                if elapsed >= max_runtime_s:
                    reason = (
                        f"运行时长已达 {args.max_runtime_minutes} 分钟 "
                        f"({elapsed / 60:.1f} 分钟)，定时终止"
                    )

            if reason:
                alert_count += 1
                log.error(
                    f"⚠  资源告警! {reason} (第 {alert_count} 次告警)"
                )

                if not args.no_kill:
                    log.warning("正在终止项目进程...")
                    kill_all_project_processes()
                    log.critical("项目进程已终止，监控退出。请释放资源后重新启动。")
                    sys.exit(1)
                else:
                    log.warning("[dry-run] 跳过杀进程")
            else:
                if alert_count > 0:
                    log.info(f"  所有资源已恢复正常，告警计数重置")
                alert_count = 0

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("监控收到中断信号，退出。")
    except Exception as e:
        log.error(f"监控异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
