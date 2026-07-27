# Архитектура

Сервис принимает задачи через FastAPI и обрабатывает их вне HTTP-запроса.
PostgreSQL хранит состояние, а RabbitMQ доставляет сообщения отдельным
worker-процессам.

Стек: Python 3.12, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, RabbitMQ и
`aio-pika`.

## Схема

```text
Client
  │
  ▼
FastAPI ──► PostgreSQL: tasks + outbox
                          │
                     dispatcher
                          │
                          ▼
                 RabbitMQ priority queue
                          │
                          ▼
                       workers
                          │
                          └──► PostgreSQL
```

PostgreSQL — источник истины для статусов и результатов. Dispatcher
публикует outbox-события, а workers выполняют заменяемый `TaskProcessor`.
API, dispatcher и workers масштабируются независимо.

## Жизненный цикл

```text
NEW → PENDING → IN_PROGRESS → COMPLETED
                   │
                   ├────────→ FAILED
                   └─ retry → PENDING

NEW / PENDING / IN_PROGRESS → CANCELLED
```

- `NEW` — задача сохранена и ожидает публикации.
- `PENDING` — ожидает worker или повторную попытку.
- `IN_PROGRESS` — выполняется worker.
- `COMPLETED`, `FAILED`, `CANCELLED` — терминальные статусы.

## Надёжность

Задача и outbox-событие создаются одной транзакцией PostgreSQL. Поэтому
принятая задача не теряется при временной недоступности RabbitMQ.

Dispatcher использует lease и publisher confirms. Worker подтверждает
сообщение только после фиксации результата в БД.

Доставка имеет гарантию at-least-once. Дубликаты обрабатываются
идемпотентно, а execution token не позволяет старому worker записать
поздний результат. Просроченный lease возвращает зависшую задачу в очередь.

Временные ошибки повторяются с ограниченным backoff. После исчерпания
попыток задача получает `FAILED`, некорректное сообщение отправляется в
dead-letter queue.

## Отмена и приоритет

`DELETE` не удаляет запись. Состояния `NEW`, `PENDING`, `IN_PROGRESS`
переходят в `CANCELLED`; повторная отмена идемпотентна. Завершённая задача
возвращает `409 Conflict`.

Отмена кооперативная: worker проверяет статус и не может перезаписать
`CANCELLED` поздним результатом.

Приоритеты `LOW`, `MEDIUM`, `HIGH` влияют на следующее сообщение, но не
прерывают уже запущенную задачу. Параллельность задаётся количеством
workers и настройкой concurrency.

## Допущения

PDF не определяет часть контракта, поэтому:

- идентификатор — UUID v4, время — UTC;
- результат и ошибка — взаимоисключающие JSON-объекты;
- фильтры списка — `status` и `priority`, пагинация — `page`/`size`;
- демонстрационный processor формирует статистику названия и описания;
- аутентификация не входит в задание, health checks служат эксплуатации.

Локальный Docker Compose не является production HA-кластером.
