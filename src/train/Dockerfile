FROM python:3.12-slim

RUN pip install uv

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync

COPY src/train/ ./train/
COPY models/ ./models/
COPY data/ ./data/

CMD [ "uv run python ./train/train_model.py" ]
