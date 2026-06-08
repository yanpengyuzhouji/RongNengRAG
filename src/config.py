"""
统一配置加载 — 自动检测项目根目录，解析相对路径
所有模块通过此模块获取配置，无需硬编码绝对路径
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional


# 缓存
_config_cache: Optional[dict] = None
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """
    自动检测项目根目录
    规则: 向上查找包含 config.yaml 的目录
    """
    global _project_root
    if _project_root is not None:
        return _project_root

    # 从当前文件位置开始向上查找
    current = Path(__file__).resolve().parent.parent  # src/ 的父级 = 项目根
    markers = ["config.yaml", "requirements.txt"]

    for _ in range(5):
        if any((current / m).exists() for m in markers):
            _project_root = current
            return _project_root
        current = current.parent

    # 回退: 当前工作目录
    _project_root = Path.cwd()
    return _project_root


def _resolve_paths(paths_dict: dict, project_root: Path) -> dict:
    """将配置中的相对路径解析为绝对路径"""
    resolved = {}
    for key, value in paths_dict.items():
        if value and isinstance(value, str):
            # 已经是绝对路径的不动
            if os.path.isabs(value):
                resolved[key] = value
            else:
                resolved[key] = str(project_root / value)
        else:
            resolved[key] = value
    return resolved


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件，自动解析相对路径
    优先从项目根目录的 config.yaml 加载
    自动加载 .env 环境变量
    """
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    project_root = get_project_root()

    # 加载 .env 文件 (如果存在)
    env_path = project_root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass  # python-dotenv 未安装时静默跳过

    # 确定配置文件路径
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = project_root / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {cfg_path}\n"
            f"请将 config.yaml 放在项目根目录: {project_root}"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 解析相对路径
    if "paths" in config:
        config["paths"] = _resolve_paths(config["paths"], project_root)

    # 存储项目根目录到 config 中供其他模块使用
    config["_project_root"] = str(project_root)

    if config_path is None:
        _config_cache = config

    return config


def get_config_path() -> str:
    """获取默认配置文件路径"""
    return str(get_project_root() / "config.yaml")


def ensure_data_dirs(config: dict = None):
    """确保数据目录存在"""
    if config is None:
        config = load_config()

    paths = config.get("paths", {})
    for key in ["uploads_dir", "parsed_cache", "models_dir", "knowledge_base"]:
        val = paths.get(key, "")
        if val and not os.path.exists(val):
            os.makedirs(val, exist_ok=True)
            print(f"[init] 创建目录: {val}")

    # milvus_db 是文件路径，确保父目录存在
    milvus = paths.get("milvus_db", "")
    if milvus:
        parent = os.path.dirname(milvus)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    # metadata_db 是文件路径
    meta = paths.get("metadata_db", "")
    if meta:
        parent = os.path.dirname(meta)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)


# 模块级快捷访问
def get(k: str, default=None):
    """快捷获取 config 中的值: get('paths.uploads_dir') 或 get('retrieval.coarse_top_k')"""
    cfg = load_config()
    keys = k.split(".")
    val = cfg
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default
