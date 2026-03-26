@echo off
:: =============================================
::  Planner 推理服务启动器 (Windows)
::  双击运行或在终端执行: start_vllm.bat [模式]
::
::  模式:
::    sglang  - SGLang FP8 量化 (默认, 推荐)
::    int8    - SGLang bitsandbytes INT8
::    ollama  - Ollama 本地部署
:: =============================================

set MODE=%1
if "%MODE%"=="" set MODE=sglang

echo ========================================
echo  Planner 推理服务启动器
echo  模式: %MODE%
echo  关闭此窗口即可停止服务
echo ========================================
echo.

if "%MODE%"=="sglang" (
    echo 服务地址: http://localhost:8000/v1
) else if "%MODE%"=="int8" (
    echo 服务地址: http://localhost:8000/v1
) else if "%MODE%"=="ollama" (
    echo 服务地址: http://localhost:11434
)

echo.
wsl bash /mnt/c/Users/sanyang/sanyangworkspace/music_recommendation/Muisc-Research/scripts/start_vllm.sh %MODE%
pause
