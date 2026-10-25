FROM python:3.12-slim

# Hugging Face Spaces runs containers as uid 1000, not root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user scoutsignal scoutsignal
COPY --chown=user app app

# Build the warehouse and train every model at image build time (seeded, ~30 s), so the
# container starts serving immediately and every replica serves identical models.
RUN python -m scoutsignal.pipeline all

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
