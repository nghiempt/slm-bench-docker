FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN python3 -m  pip install --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/output

COPY gpu_monitor.py .
COPY main.py .
COPY main.json .
# COPY test.json .
COPY run.py .

CMD ["python3", "run.py"]
