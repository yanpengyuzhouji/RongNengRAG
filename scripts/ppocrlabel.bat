@echo off
REM PPOCRLabel 启动脚本 — 榕能RAG数据标注工具
REM 独立虚拟环境，不影响项目 OCR 引擎

cd /d "%~dp0..\tools\PPOCRLabel"

REM 使用独立 venv 的 Python（paddlepaddle-gpu 3.2.2 + paddleocr 3.6.0）
set PYTHON=.venv\Scripts\python.exe

REM 跳过模型下载网络检测
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo ==========================================
echo   PPOCRLabel 数据标注工具 v2
echo   榕能电力审图知识库 RAG
echo   GPU: NVIDIA RTX 4070 (CUDA 11.8)
echo ==========================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] 虚拟环境未安装，请先运行:
    echo   cd tools\PPOCRLabel
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install pyqt5 paddleocr
    echo   .venv\Scripts\pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    pause
    exit /b 1
)

echo 首次运行需下载模型 (~500MB)，请耐心等待...
echo.

%PYTHON% PPOCRLabel.py --gpu True --lang ch %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] PPOCRLabel 启动失败
    pause
)
