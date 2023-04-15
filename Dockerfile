FROM python:3.10.10

WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /src

ENV PYTHONPATH=/src

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
