FROM solo12-scanweave:1.0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV LENSBENCH_DATA_DIR=/data PORT=18921
EXPOSE 18921
CMD ["python3", "app.py"]
