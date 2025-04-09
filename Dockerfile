FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
