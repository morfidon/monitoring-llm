@echo off
echo Setting up Real LLM Demo with Docker
echo ====================================

echo.
echo 1. Checking if Docker is running...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo Docker is running!

echo.
echo 2. Setting OpenAI API key...
set /p OPENAI_KEY="Enter your OpenAI API key (or press Enter to skip): "
if not "%OPENAI_KEY%"=="" (
    set OPENAI_API_KEY=%OPENAI_KEY%
    echo API key set for this session
) else (
    echo No API key set - will use simulation mode
)

echo.
echo 3. Building and starting containers...
docker-compose -f docker-compose-real.yml down
docker-compose -f docker-compose-real.yml up -d --build

echo.
echo 4. Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo 5. Checking service status...
curl -s http://localhost:8000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo LLM App: RUNNING (http://localhost:8000)
) else (
    echo LLM App: Starting...
)

curl -s http://localhost:9090/ >nul 2>&1
if %errorlevel% equ 0 (
    echo Prometheus: RUNNING (http://localhost:9090)
) else (
    echo Prometheus: Starting...
)

curl -s http://localhost:3000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo Grafana: RUNNING (http://localhost:3000)
) else (
    echo Grafana: Starting...
)

echo.
echo ====================================
echo Real LLM Demo is ready!
echo.
echo Services:
echo - LLM App: http://localhost:8000
echo - Prometheus: http://localhost:9090
echo - Grafana: http://localhost:3000 (admin/admin)
echo.
echo Next steps:
echo 1. Run 'python real-llm-demo.py' for real API demo
echo 2. Run 'python demo.py' for simulated demo
echo 3. View dashboard at http://localhost:3000
echo.
echo To stop: docker-compose -f docker-compose-real.yml down
echo ====================================
pause
