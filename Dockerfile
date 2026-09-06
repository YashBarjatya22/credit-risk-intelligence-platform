FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/runtime /app/data && chown -R appuser:appuser /app/runtime /app/data
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8501')+'/_stcore/health',timeout=3)"
CMD ["sh", "-c", "python bootstrap.py && python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
