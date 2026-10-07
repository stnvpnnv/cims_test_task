# cims_test_task

## Запуск

Скопируйте `.env.example` в `.env`, задайте пароли PostgreSQL и RabbitMQ, затем
заполните оба `CIMS_INTERNAL_*_URL`. Пароли внутри URL должны быть
percent-encoded независимо от исходных значений.

```shell
docker compose up --build -d
```

Compose запускает API, dispatcher и worker; документация API по умолчанию
доступна на `http://localhost:8000/docs`, интерфейс RabbitMQ — на
`http://localhost:15672`. Состояние сервисов показывает `docker compose ps`.
Остановка через `docker compose down` сохраняет данные в именованных volumes.

## Создание задачи

`POST /api/v1/tasks` принимает `name`, `description` и приоритет `LOW`, `MEDIUM`
или `HIGH`. Новая задача возвращается с кодом `201` и заголовком `Location`.

Необязательный `Idempotency-Key` — регистрозависимый HTTP token длиной до 255
символов; рекомендуется случайный UUID. Ключ действует всё время хранения задачи.
Повтор того же тела возвращает текущее состояние задачи с кодом `200`, а тот же
ключ с другим телом — `422 application/problem+json`. Без заголовка каждый запрос
создаёт новую задачу.

## Получение задач

`GET /api/v1/tasks` возвращает страницу задач; фильтры `status` и `priority`
объединяются через AND. По умолчанию `page=1`, `size=20` (максимум 100);
сначала возвращаются новые задачи (`created_at DESC`, затем `id DESC`).

`GET /api/v1/tasks/{task_id}` принимает UUID v4 и возвращает полное текущее
состояние задачи. Неизвестный идентификатор даёт `404 application/problem+json`,
а некорректный — стандартный `422`.

Для частого опроса `GET /api/v1/tasks/{task_id}/status` возвращает только
идентификатор и текущий статус; правила идентификатора и ошибок те же.

## Отмена задачи

`DELETE /api/v1/tasks/{task_id}` принимает UUID v4 без тела запроса и переводит
`NEW`, `PENDING` или `IN_PROGRESS` в `CANCELLED`. В ответ возвращается полное
текущее представление `TaskResponse` с кодом `200`. Повторная отмена также
возвращает `200` и не изменяет исходный `finished_at`; отдельный
`Idempotency-Key` не нужен.

Неизвестная задача даёт `404 application/problem+json`, а задача в состоянии
`COMPLETED` или `FAILED` — `409 application/problem+json`. Некорректный
идентификатор возвращает стандартный `422`.

## Проверки

Установить зависимости: `poetry install`. Локальные проверки без внешних сервисов:

```shell
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest -m "not integration"
```

Интеграционные тесты операций с задачами требуют PostgreSQL и отдельной
существующей базы с именем, оканчивающимся на `_test`. Пользователю БД нужны права
подключения и создания схем. RabbitMQ для database-тестов не нужен.

Задайте `CIMS_TEST_DATABASE_URL` в окружении по примеру из `.env.example`,
подставив пароль с URL-кодированием. Python не загружает `.env` автоматически.
Параметры query в тестовом URL не допускаются. Затем выполните:

```shell
poetry run pytest tests/integration/database --no-cov
```

Каждый тест применяет миграции Alembic в своей случайной схеме `cims_test_*`,
а при завершении удаляет только эту схему вместе с её объектами. Тестовые
соединения не используют схему `public`. При аварийном завершении процесса
схема может остаться в тестовой БД.

Тесты publisher и consumer требуют отдельного существующего RabbitMQ vhost с именем,
оканчивающимся на `_test`. Задайте `CIMS_TEST_RABBITMQ_URL` с явными учётными
данными и без query или fragment; пользователю нужны права configure, write и read:

```shell
poetry run pytest tests/integration/messaging --no-cov
```

Тесты создают exchange и queue с уникальными именами и удаляют только свои ресурсы.
Сквозной сценарий требует обе тестовые переменные и проверяет API через ASGI transport,
реальные PostgreSQL и RabbitMQ, dispatcher и worker до результата `COMPLETED`:

```shell
poetry run pytest tests/integration/test_task_pipeline.py --no-cov
```

Соответствующая группа интеграционных тестов пропускается с пояснением,
если её `CIMS_TEST_*_URL` не задана; некорректный URL или недоступный сервис
приводят к ошибке. Для полного прогона с покрытием задайте обе тестовые переменные
и выполните `poetry run pytest`.

## CI

GitHub Actions запускает проверки при push, pull request и вручную:

- Ruff, форматирование, pre-commit, mypy и согласованность `poetry.lock`;
- все тесты с отдельными PostgreSQL/RabbitMQ, без пропусков и с покрытием не ниже 90%;
- сборку runtime Docker-образа после успешных проверок, без публикации и развёртывания.

CI не использует локальный `.env` или рабочую БД. Отчёты JUnit и покрытия доступны
в артефактах запуска в течение 7 дней. Workflow: [ci.yml](.github/workflows/ci.yml).
