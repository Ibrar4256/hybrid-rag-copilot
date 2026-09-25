FROM python:3.10-slim

WORKDIR /app

# CPU-only torch first (matches README's Setup step) — avoids pulling ~1.5GB
# of unneeded CUDA packages that requirements.txt's transitive deps would
# otherwise resolve to on a GPU-less container.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "research_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
