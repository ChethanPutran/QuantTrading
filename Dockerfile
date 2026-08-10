FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY uv.lock /app/
COPY src /app/src
COPY main.py main_async.py /app/

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --extra full

ENTRYPOINT ["uv", "run", "trading-system"]
CMD ["--steps", "500"]