FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=user . .

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
