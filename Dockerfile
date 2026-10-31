# Bootstrap container only. Benchmark CLI will be added in milestone M1.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev
CMD ["uv", "run", "--locked", "python", "-c", "print('Scaffold only: benchmark CLI is not implemented yet')"]
