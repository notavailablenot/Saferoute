# SafeRoute backend: FastAPI + ONNX Runtime (CPU), fully offline once built.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SAFEROUTE_DB_URL=sqlite:////app/db/saferoute.db \
    SAFEROUTE_CLS_MODEL=/app/models/baseline_cnn.onnx \
    SAFEROUTE_DET_MODEL=/app/models/detector_yolo11n.onnx \
    SAFEROUTE_CLASSES=/app/configs/classes.yaml

WORKDIR /app

# libgomp1: OpenMP runtime used by ONNX Runtime / OpenCV wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# dependencies first so code changes do not invalidate the pip layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY saferoute/ saferoute/
COPY configs/classes.yaml configs/
COPY models/*.onnx models/

# run as a non-root user; the database lives in a named volume at /app/db
RUN useradd --create-home --uid 1000 saferoute && mkdir -p /app/db && chown -R saferoute /app/db
USER saferoute

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read(); sys.exit(0 if b'\"ready\"' in r else 1)"

CMD ["uvicorn", "saferoute.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
