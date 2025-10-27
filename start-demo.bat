@echo off
echo 🚀 Starting LLM Hallucination Monitoring Demo
echo =============================================

echo.
echo 📦 Starting Docker containers...
docker-compose up -d

echo.
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo 🔍 Checking service health...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ LLM App is running
) else (
    echo ❌ LLM App is not responding
)

curl -s http://localhost:9090 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus is running
) else (
    echo ❌ Prometheus is not responding
)

curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Grafana is running
) else (
    echo ❌ Grafana is not responding
)

echo.
echo 🎯 Running demo traffic...
python demo.py

echo.
echo 📊 Demo completed! You can now view:
echo    Grafana Dashboard: http://localhost:3000 (admin/admin)
echo    Prometheus: http://localhost:9090
echo    LLM App: http://localhost:8000
echo.

pause
