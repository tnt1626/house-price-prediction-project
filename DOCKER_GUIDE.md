# Docker Setup Guide - Hướng dẫn thiết lập Docker

## Tổng quan / Overview

Dự án này đã được cấu hình để chạy với Docker, bao gồm:
- **API Service**: FastAPI backend chạy trên port 8000
- **Frontend Service**: Streamlit app chạy trên port 8501

## Yêu cầu / Requirements

1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)

Kiểm tra cài đặt:
```bash
docker --version
docker-compose --version
```

## Cách sử dụng / How to Use

### 1. Build Images

Xây dựng Docker images cho toàn bộ ứng dụng:

```bash
docker-compose build
```

Hoặc build chỉ một service:
```bash
docker-compose build api      # Chỉ API
docker-compose build frontend # Chỉ Frontend
```

### 2. Chạy ứng dụng / Run Application

Khởi động cả API và Frontend:

```bash
docker-compose up
```

Chạy ở background:
```bash
docker-compose up -d
```

Khởi động một service cụ thể:
```bash
docker-compose up api      # Chỉ API
docker-compose up frontend # Chỉ Frontend
```

### 3. Dừng ứng dụng / Stop Application

```bash
# Dừng tất cả containers
docker-compose down

# Dừng và xóa volumes
docker-compose down -v

# Dừng một service cụ thể
docker-compose stop api
```

### 4. Xem logs / View Logs

```bash
# Xem logs từ tất cả services
docker-compose logs -f

# Xem logs từ API service
docker-compose logs -f api

# Xem logs từ Frontend service
docker-compose logs -f frontend

# Xem 100 dòng log cuối cùng
docker-compose logs --tail=100
```

### 5. Chạy Commands trong Container

```bash
# Chạy command trong API container
docker-compose exec api python --version

# Chạy command trong Frontend container
docker-compose exec frontend ls -la

# Truy cập shell
docker-compose exec api /bin/bash
docker-compose exec frontend /bin/bash
```

## Truy cập ứng dụng / Access Application

Khi containers đang chạy:

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **API Redoc**: http://localhost:8000/redoc
- **API Health Check**: http://localhost:8000/health
- **Streamlit Frontend**: http://localhost:8501

## Cấu hình / Configuration

### Các tệp quan trọng:

1. **Dockerfile**: Định nghĩa cách xây dựng image
   - Multi-stage build để giảm kích thước image
   - Sử dụng Python 3.11-slim
   - Cài đặt tất cả dependencies từ requirements.txt

2. **docker-compose.yml**: Orchestration cho các services
   - API service: FastAPI server
   - Frontend service: Streamlit app
   - Healthcheck cho API
   - Volumes cho data persistence

3. **.dockerignore**: Loại bỏ các file không cần thiết khỏi build
   - __pycache__, .git, venv, notebooks, logs, v.v.

### Biến môi trường / Environment Variables

API service:
- `API_HOST=0.0.0.0`
- `API_PORT=8000`

Frontend service:
- `API_URL=http://api:8000` (kết nối tới API service)
- `STREAMLIT_SERVER_HEADLESS=true`

Để thay đổi cấu hình, chỉnh sửa `docker-compose.yml` trong phần `environment`.

## Quản lý Volumes

### Data Persistence

Các volumes được mount:
```yaml
api:
  volumes:
    - ./artifacts:/app/artifacts:ro        # Models (read-only)
    - ./logs:/app/logs                      # Logs
    - ./data:/app/data:ro                   # Training data (read-only)

frontend:
  volumes:
    - ./src/frontend:/app/src/frontend      # Frontend code (live updates)
    - ./artifacts:/app/artifacts:ro
    - ./data:/app/data:ro
```

- `:ro` = read-only mount
- `:rw` hoặc không ghi = read-write mount

### Xóa Volumes

```bash
# Xóa tất cả volumes liên quan
docker-compose down -v
```

## Troubleshooting

### Port đã được sử dụng / Port already in use

Nếu port 8000 hoặc 8501 đã được sử dụng:

```bash
# Thay đổi port trong docker-compose.yml
# Ví dụ: "8002:8000" thay vì "8000:8000"

# Hoặc tìm process sử dụng port (Linux/Mac)
lsof -i :8000
lsof -i :8501

# Windows
netstat -ano | findstr :8000
```

### Container không khởi động được / Container won't start

```bash
# Kiểm tra logs
docker-compose logs api
docker-compose logs frontend

# Xóa cache và build lại
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Health check thất bại / Health check failed

```bash
# Kiểm tra API container
docker-compose exec api curl http://localhost:8000/health

# Xem logs
docker-compose logs api
```

### Lỗi module không tìm thấy / Module import errors

Đảm bảo:
1. `requirements.txt` được cập nhật đúng
2. Rebuild image: `docker-compose build --no-cache`
3. Kiểm tra PYTHONPATH trong dockerfile

## Production Deployment

Để deploy lên production:

1. **Sử dụng specific image tags**:
   ```yaml
   image: house-price-prediction:1.0.0
   ```

2. **Sử dụng environment file**:
   ```bash
   docker-compose --env-file .env.production up -d
   ```

3. **Sử dụng Reverse Proxy** (nginx):
   - API trên port 8000
   - Frontend trên port 8501
   - Sử dụng nginx để route requests

4. **Scaling** (Docker Swarm hoặc Kubernetes):
   - Có thể scale thêm API instances
   - Sử dụng load balancer

## Một số lệnh hữu ích / Useful Commands

```bash
# Xem tất cả containers
docker ps -a

# Xem usage disk
docker system df

# Dọn dẹp unused resources
docker system prune -a

# Build với progress output
docker-compose build --progress=plain

# Validate docker-compose file
docker-compose config

# Scale services
docker-compose up -d --scale api=3

# Kiểm tra network
docker network ls
docker network inspect house-price-prediction_house-price-network
```

## Support & Documentation

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [Streamlit Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
