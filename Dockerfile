# استخدام صورة بايثون رسمية خفيفة
FROM python:3.10-slim

# تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# تثبيت المتطلبات الأساسية للنظام إن أمكن
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات أولاً للاستفادة من الكاش (Caching)
COPY requirements.txt .

# تثبيت مكتبات البايثون
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى داخل الحاوية
COPY . .

# تعيين البورت الافتراضي (Railway بيحدد البورت ديناميكياً عبر المتغير PORT)
ENV PORT=8000

# أمر تشغيل السيرفر باستخدام Uvicorn ودعم متغير PORT للـ Cloud
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}