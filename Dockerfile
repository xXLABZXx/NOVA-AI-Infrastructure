# Sandbox image: untrusted network fetches happen HERE, never on the host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Least privilege: run as non-root.
RUN useradd -m -u 10001 nova && mkdir -p /app /quarantine /logs \
    && chown -R nova:nova /app /quarantine /logs

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .

# No secrets baked into the image. Pass at runtime: --env-file .env
USER nova

# Default: run the sandbox fetch worker (see scripts/). Override as needed.
CMD ["python", "-m", "nova.sandbox.worker"]
