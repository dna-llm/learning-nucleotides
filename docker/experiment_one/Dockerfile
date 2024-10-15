# Build stage
FROM python:3.11 AS builder

# Install uv
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Set working directory
WORKDIR /app
# Copy lockfile
COPY ../../uv.lock .
COPY ../../pyproject.toml .
COPY ../../README.md .
# Create venv and install dependencies
RUN /root/.cargo/bin/uv sync --no-build-isolation-package flash-attn --frozen
COPY ../../.venv /app/.venv
COPY ../../experiment_one /app/experiment_one

# Final stage
FROM python:3.11-slim
# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# Copy virtual environment and application code from builder
COPY --from=builder /app /app
# Set working directory
WORKDIR /app
# Set entrypoint
ENTRYPOINT ["python", "run_hf_model.py"]
