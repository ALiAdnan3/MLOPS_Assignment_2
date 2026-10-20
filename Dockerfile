# VisionGuard web app (Task 6.2)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when requirements-serve.txt changes,
#    so editing the source code does not re-download ~400 MB of packages.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) unprivileged user (a compromised app cannot modify the image or run as root)
RUN useradd --create-home --uid 10001 appuser

# 3) source code last - changes most often
COPY visionguard/ visionguard/
COPY webapp/ webapp/

USER appuser
EXPOSE 8000
# Docker marks the container unhealthy if /health stops answering (model or DB broken)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "webapp.app"]
