# Data Cleaning Pipeline OpenEnv

Production-ready OpenEnv benchmark environment for data cleaning agents.

## Tasks

1. `orders` (easy)
2. `user-merge` (medium)
3. `transactions` (hard)

## Project Layout

```text
.
├── app.py
├── openenv.yaml
├── inference.py
├── validate.py
├── env/
│   ├── server.py
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   └── actions.py
├── data/
│   ├── generate.py
│   ├── orders_dirty.csv
│   ├── users_dirty.csv
│   └── transactions_dirty.csv
├── graders/
│   ├── orders_grader.py
│   ├── users_grader.py
│   └── transactions_grader.py
└── tests/
    ├── test_env.py
    └── test_graders.py
```

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m data.generate
uvicorn env.server:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t data-cleaning-openenv .
docker run -p 8000:8000 data-cleaning-openenv
```

With judge-like constraints:

```bash
docker run --cpus=2 --memory=8g -p 8000:8000 data-cleaning-openenv
```

## Endpoints

1. `GET /` health check
2. `POST /reset` body: `{"task_id": "orders"}`
3. `POST /step?task_id=orders` body: `Action`
4. `GET /state?task_id=orders`
5. `GET /tasks`

## Run Inference Baseline

```bash
export ENV_URL="http://localhost:8000"
python inference.py
```

## Validation

```bash
pytest -q
python validate.py
openenv validate
```

## Push To GitHub

```bash
git init
git add .
git commit -m "feat: productionize openenv data cleaning benchmark"
git branch -M main
git remote add origin https://github.com/<your-username>/data-cleaning-openenv.git
git push -u origin main
```

## Deploy To Hugging Face Space

1. Create a new Space with SDK = Docker.
2. Set Space visibility and hardware.
3. Push repository to the Space remote:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/data-cleaning-openenv
git push hf main
```

4. After build, test:

```bash
curl https://<your-username>-data-cleaning-openenv.hf.space/
```
