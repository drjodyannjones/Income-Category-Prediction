FROM python:3.10.10

WORKDIR /src/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.app.main:app", "--bind", "0.0.0.0:8000"]