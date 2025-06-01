FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY . /app
WORKDIR /app
RUN uv sync --locked --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
EXPOSE 8052
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "--server.port", "8052", "--server.fileWatcherType", "none", "app.py"]
