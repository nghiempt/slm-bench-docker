docker build -t slm .
docker run -e WANDB_API_KEY=c8bb68d3bbc09627eb5abd2a786528aff7eb5103 --runtime nvidia --rm -v $(pwd)/output:/app/output slm
