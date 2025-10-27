# Setup Instructions for LLM Monitoring Demo

## Prerequisites
1. **Docker Desktop must be running** - This is the most common issue
2. Python 3.11+ installed

## Step 1: Start Docker Desktop
- Open Docker Desktop from your Start Menu
- Wait for it to fully start (green icon in system tray)
- Verify it's working by running: `docker ps`

## Step 2: Deploy the Monitoring Stack
```bash
docker-compose up -d
```

## Step 3: Wait for Services to Start
The services need 30-60 seconds to fully initialize. You can check progress with:
```bash
docker-compose logs -f
```

## Step 4: Run the Demo
```bash
python demo.py
```

## Troubleshooting

### If Docker Desktop won't start:
1. Restart your computer
2. Make sure virtualization is enabled in BIOS
3. Check Windows Features for "Hyper-V" and "Windows Subsystem for Linux"

### If services fail to start:
```bash
# Check what's running
docker-compose ps

# View logs
docker-compose logs llm-app
docker-compose logs prometheus
docker-compose logs grafana

# Reset everything
docker-compose down
docker-compose up -d
```

### If ports are already in use:
```bash
# Check what's using the ports
netstat -ano | findstr :8000
netstat -ano | findstr :9090
netstat -ano | findstr :3000

# Kill the processes using the ports
taskkill /PID <PID> /F
```

## Expected URLs After Startup
- LLM App: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Metrics: http://localhost:9464/metrics

## Quick Verification
Once Docker Desktop is running, you should be able to run:
```bash
docker ps
```
And see containers starting up.
