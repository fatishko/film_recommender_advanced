# Rəsmi Python image-dən başlayırıq
FROM python:3.10

# İş mühiti (container içində)
WORKDIR /app

# Lazım olan faylları əlavə edirik
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model və app faylını əlavə et
COPY . .

# Port (Gradio üçün)
EXPOSE 7860

# Proqramı işə sal
CMD ["python", "app/app.py"]
