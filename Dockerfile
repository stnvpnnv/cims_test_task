# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.14

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG POETRY_VERSION=2.1.3

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --only main --no-root --no-directory --no-ansi

COPY src ./src

RUN poetry build --format wheel --clean \
    && .venv/bin/python -m pip install --no-cache-dir --no-deps dist/*.whl \
    && .venv/bin/python -m pip check


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV VIRTUAL_ENV=/app/.venv

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid "${APP_GID}" cims \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin cims

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml ./
COPY migrations ./migrations

USER ${APP_UID}:${APP_GID}

EXPOSE 8000

STOPSIGNAL SIGTERM

CMD ["python", "-m", "uvicorn", "cims_task_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
