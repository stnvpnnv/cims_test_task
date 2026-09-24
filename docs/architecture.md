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

Очередь выполнения и dead-letter queue объявляются как durable quorum queues
RabbitMQ 4.3+. Dispatcher передаёт `LOW`, `MEDIUM`, `HIGH` как приоритеты `1`,
`2`, `3`. Имена очередей версионируются, поскольку их тип и dead-letter
аргументы нельзя безопасно изменить повторным объявлением.

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

`COUNT(*)` и выборка страницы выполняются в короткой транзакции
`REPEATABLE READ`, поэтому результат формируется из одного снимка данных.

Dispatcher использует lease и publisher confirms. Worker подтверждает
сообщение только после фиксации результата в БД.

Процесс dispatcher управляет циклами публикации outbox и восстановления
просроченных execution lease. Они используют общий сигнал остановки и пул БД;
сбой одного цикла останавливает второй. Одно соединение резервируется recovery:
`database_pool_size + database_max_overflow >= dispatcher_batch_size + 1`.

Publisher отправляет persistent-сообщение как mandatory и считает публикацию
успешной только после подтверждения RabbitMQ; `Basic.Return` является ошибкой.
`message_id` равен UUID outbox-события, `correlation_id` — UUID задачи и не меняются
при повторной публикации. Это позволяет consumer'у дедуплицировать сообщения, но
не превращает at-least-once в exactly-once. Publisher отвечает только за broker I/O;
lease, фиксация `published_at` и планирование retry принадлежат dispatcher'у.

Основной путь доставки имеет гарантию at-least-once. Дубликаты обрабатываются
идемпотентно, а execution token не позволяет старому worker записать
поздний результат. Просроченный lease возвращает зависшую задачу в очередь.

Временные ошибки повторяются с ограниченным backoff. После исчерпания
попыток задача получает `FAILED`, некорректное сообщение отправляется в
dead-letter queue.

Стандартный перенос в DLQ остаётся at-most-once и служит диагностике;
PostgreSQL сохраняет роль источника истины.

Плановый retry не использует TTL или немедленный requeue: время следующей
публикации хранится в `outbox.available_at`, чтобы PostgreSQL оставался
единственным источником состояния повторов.

## Отмена и приоритет

`DELETE` не удаляет запись. Одним условным `UPDATE` состояния `NEW`, `PENDING`
и `IN_PROGRESS` переходят в `CANCELLED`; повторная отмена возвращает ту же задачу
без изменения `finished_at`. Для `COMPLETED` и `FAILED` API возвращает
`409 Conflict`.

Изменение задачи и погашение её неопубликованных outbox-событий выполнения
проводятся одной транзакцией с порядком блокировок task → outbox.
`clock_timestamp()` задаёт фактическое время отмены даже после ожидания
конкурентной транзакции. Отмена очищает `dispatch_token`, `execution_token`
и lease.

Уже опубликованное сообщение отозвать из RabbitMQ нельзя, поэтому отмена
кооперативная. Worker должен фиксировать результат условным `UPDATE` по
`status = IN_PROGRESS` и `execution_token`. Это fencing-условие не позволит
проигравшему гонку worker'у перезаписать `CANCELLED` поздним результатом; начатый
внешний побочный эффект может потребовать отдельной компенсации.

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
