FROM python:3.11-slim

# system deps (fast NumPy/Pillow wheels avoid dev headers)
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8080 PYTHONUNBUFFERED=1
ENV DEBUG=0
CMD ["gunicorn", "bumpserver.wsgi:application", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "60"]
