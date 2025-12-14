FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1 \
      libglib2.0-0 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      libsm6 \
      xvfb \
      xauth \
    && rm -rf /var/lib/apt/lists/*

# install Python deps (editable install so local code changes are reflected if you mount a volume)
COPY requirements.txt setup.py pyproject.toml README.md /app/
COPY pymarlzooplus /app/pymarlzooplus

RUN mkdir -p /app/pymarlzooplus/results

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .

# run the experiment entrypoint; pass args via `docker run ... --config=... --env-config=... with ...`
ENTRYPOINT ["python", "pymarlzooplus/main.py"]