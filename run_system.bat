@echo off
TITLE He Thong Tim Kiem Phan Tan (Dask Cluster) - UNLIMITED RAM

:: 1. Cấu hình chung
set "WORK_DIR=%cd%"
set "VENV_PATH=%WORK_DIR%\venv\Scripts\activate.bat"
set "PYTHONPATH=%WORK_DIR%\app\src"
set "SCHEDULER=tcp://127.0.0.1:8786"

:: Kiểm tra venv
if not exist "%VENV_PATH%" (
    echo [LOI] Khong tim thay venv! Hay tao venv truoc.
    pause
    exit
)

echo [1/3] Dang khoi dong Scheduler...
start "Dask Scheduler" cmd /k "call %VENV_PATH% && dask-scheduler --host 127.0.0.1"
timeout /t 5 >nul

echo [2/3] Dang khoi dong Service API...
start "Query Service" cmd /k "call %VENV_PATH% && set PYTHONPATH=%PYTHONPATH%&& set META_PATH=data/meta.parquet&& set MINHASH_META=data/minhash_meta.pkl&& set SHARD_SIZE=5000&& set DASK_SCHEDULER_ADDRESS=%SCHEDULER%&& uvicorn app.src.query_service:app --host 127.0.0.1 --port 8000"
timeout /t 5 >nul

echo [3/3] Dang khoi dong Workers (CHE DO KHONG GIOI HAN RAM)...

:: --- WORKER 1 ---
:: Them co: --memory-limit 0 (Khong gioi han RAM)
:: Them co: --no-nanny (Tat giam sat de khong bi kill khi CPU 100%)
start "Worker 1 (Main)" cmd /k "call %VENV_PATH% && set PYTHONPATH=%PYTHONPATH%&& set SHARD_DIR=data/shards&& set DASK_SCHEDULER_ADDRESS=%SCHEDULER%&& dask-worker %SCHEDULER% --nthreads 1 --no-nanny --memory-limit 0 --name worker-1"

:: --- WORKER 2 ---
:: Neu may ban co > 16GB RAM thi hay bat Worker 2. 
:: Neu may yeu (8GB RAM), chay 2 worker se bi lag may do moi con an 4.7GB -> Tong 9.5GB.
:: Toi tam thoi REM (an) dong duoi di de chay 1 Worker cho an toan truoc.
:: Muon chay thi xoa chu "REM " o dau dong di.

REM timeout /t 2 >nul
REM start "Worker 2" cmd /k "call %VENV_PATH% && set PYTHONPATH=%PYTHONPATH%&& set SHARD_DIR=data/shards&& set DASK_SCHEDULER_ADDRESS=%SCHEDULER%&& dask-worker %SCHEDULER% --nthreads 1 --no-nanny --memory-limit 0 --name worker-2"

echo.
echo ===================================================
echo   HE THONG DANG KHOI DONG (CHẾ ĐỘ RAM THOẢI MÁI)!
echo   Luu y: Worker se an khoang 4-5GB RAM.
echo   Hay cho khoang 1 phut de Load xong du lieu.
echo ===================================================
pause