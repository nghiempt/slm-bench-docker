FROM python:3.9-slim

WORKDIR /app

COPY main.py .
COPY main.json .
# COPY test.json .
COPY run.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/output

CMD ["python", "run.py"]