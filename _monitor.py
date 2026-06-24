#!/usr/bin/env python
"""后台监控：C盘空间 + 端口7860/8000 存活检测"""
import time, shutil, socket, sys, os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_monitor_log.txt")
CHECK_INTERVAL = 30  # 秒
DISK_ALERT_GB = 5.0  # 低于此值告警
DISK_CRITICAL_GB = 2.0  # 低于此值严重告警
DISK_KILL_GB = 1.0     # 低于此值自动杀死8000端口进程

PORTS = [7860, 8000]
KILL_PORT = 8000  # 空间不足时杀死的端口

def check_port(port):
    """检查端口是否在监听"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except:
        return False

def check_disk():
    """返回 (剩余GB, 总GB)"""
    usage = shutil.disk_usage('C:/')
    return usage.free / (1024**3), usage.total / (1024**3)

def get_pid_by_port(port):
    """通过netstat获取端口对应的PID"""
    import subprocess
    try:
        out = subprocess.check_output(
            f'netstat -ano | findstr ":{port} " | findstr "LISTENING"',
            shell=True, text=True, timeout=5
        )
        for line in out.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5:
                return int(parts[-1])
    except:
        pass
    return None

def kill_process(pid, port):
    """强制杀死进程"""
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        if not check_port(port):
            return True, "SIGTERM"
        # 如果没死，用taskkill强制
        os.system(f'taskkill /F /PID {pid} 2>nul')
        time.sleep(1)
        if not check_port(port):
            return True, "taskkill /F"
        return False, "进程未响应"
    except Exception as e:
        return False, str(e)

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "[OK]", "WARN": "[WARN]", "ERROR": "[CRIT]"}.get(level, "")
    line = f"[{timestamp}] {prefix} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    log(f"监控已启动 | 间隔={CHECK_INTERVAL}s | 告警={DISK_ALERT_GB}GB | 严重={DISK_CRITICAL_GB}GB | 自毁={DISK_KILL_GB}GB(杀端口{KILL_PORT})")

    prev_free = None
    killed = False  # 防止重复杀

    while True:
        # --- C盘空间 ---
        free_gb, total_gb = check_disk()
        pct = (1 - free_gb/total_gb) * 100

        if prev_free is not None:
            delta = free_gb - prev_free
            delta_str = f" 变化: {delta:+.2f} GB"
        else:
            delta_str = ""

        if free_gb < DISK_CRITICAL_GB:
            level = "ERROR"
        elif free_gb < DISK_ALERT_GB:
            level = "WARN"
        else:
            level = "INFO"

        log(f"C盘剩余: {free_gb:.2f} GB / {total_gb:.0f} GB ({pct:.1f}%已用){delta_str}", level)
        prev_free = free_gb

        # --- 自动保护: C盘低于1GB时杀死8000端口 ---
        if free_gb < DISK_KILL_GB and not killed:
            log(f"!!! C盘低于 {DISK_KILL_GB} GB !!! 执行自动保护: 杀死端口 {KILL_PORT}", "ERROR")
            pid = get_pid_by_port(KILL_PORT)
            if pid:
                log(f"端口 {KILL_PORT} 对应 PID={pid}, 正在杀死...", "ERROR")
                success, method = kill_process(pid, KILL_PORT)
                if success:
                    log(f"!!! 已通过 {method} 杀死 PID={pid} (端口{KILL_PORT}) !!! C盘保护已触发", "ERROR")
                    killed = True
                else:
                    log(f"!!! 杀死 PID={pid} 失败: {method} !!! 请手动处理!", "ERROR")
            else:
                log(f"端口 {KILL_PORT} 未找到进程，无需杀死", "WARN")
                killed = True  # 避免重复查

        # --- 端口存活 ---
        for port in PORTS:
            alive = check_port(port)
            if not alive:
                if port == KILL_PORT and killed:
                    log(f"端口 {port} 已被保护性关闭", "WARN")
                else:
                    log(f"端口 {port} 不在监听!", "ERROR")
            elif int(time.time()) % (CHECK_INTERVAL * 10) < CHECK_INTERVAL:
                log(f"端口 {port} 正常", "INFO")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
