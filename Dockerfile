FROM python:3.10.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/app /app

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
