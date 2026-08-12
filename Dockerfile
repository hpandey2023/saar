FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    SAAR_CONFIG=/config/saar.yaml

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir . && \
    mkdir -p /config /data/source /state /home/app/.cache && \
    chown -R app:app /config /data /state /home/app

USER app
EXPOSE 8000
ENTRYPOINT ["saar"]
CMD ["serve"]
