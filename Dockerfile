FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "ck2milvusv2", "--help"]
