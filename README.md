# ---- LOCAL ---- # Nếu chạy trực tiếp trên máy
## (1) Build Image
docker build -t slm-bench .

## (2) Run Image
docker run --rm -v $(pwd)/output:/app/output slm-bench

--------------------------------

# ---- HUB ---- # Nếu push lên Docker Hub
## (1) Build Image in Docker Hub
docker build -t nghiempt/slm-bench .

## (2) Push Image
docker push nghiempt/slm-bench:latest

## (3) Pull Image
docker pull nghiempt/slm-bench

## (4) Run Image
docker run -v $(pwd)/output:/app/output nghiempt/slm-bench

--------------------------------

# Dockerfile dòng số 7. Có thể chạy với file test.json (5 times) để thử xem chạy và lưu kết quả được không?