"""Real-time streaming simulation for sensor data.

Implements a Kafka-style mock using asyncio queues to simulate
real-time sensor data arrival with configurable throughput.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class StreamMessage:
    """A single message in the sensor data stream."""

    engine_id: int
    cycle: int
    timestamp: str
    sensors: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "engine_id": self.engine_id,
                "cycle": self.cycle,
                "timestamp": self.timestamp,
                "sensors": self.sensors,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> StreamMessage:
        parsed = json.loads(data)
        return cls(**parsed)


class StreamProducer:
    """Produces sensor data messages to an async queue.

    Simulates a Kafka producer by reading from a DataFrame
    and pushing records at configurable intervals.
    """

    def __init__(
        self,
        queue: asyncio.Queue[StreamMessage],
        records_per_second: float = 10.0,
    ) -> None:
        self._queue = queue
        self._delay = 1.0 / records_per_second
        self._running = False

    async def produce_from_dataframe(self, df: pd.DataFrame) -> None:
        """Stream records from a DataFrame into the queue."""
        self._running = True
        logger.info(
            "Starting stream producer with {} records", len(df)
        )

        skip_cols = {"engine_id", "cycle", "timestamp"}
        for _, row in df.iterrows():
            if not self._running:
                break

            sensors = {
                col: float(row[col])
                for col in df.columns
                if col not in skip_cols and pd.notna(row[col])
            }

            message = StreamMessage(
                engine_id=int(row["engine_id"]),
                cycle=int(row["cycle"]),
                timestamp=str(
                    row.get("timestamp", datetime.now(tz=UTC).isoformat())
                ),
                sensors=sensors,
                metadata={"produced_at": datetime.now(tz=UTC).isoformat()},
            )

            await self._queue.put(message)
            await asyncio.sleep(self._delay)

        logger.info("Stream producer finished")

    def stop(self) -> None:
        self._running = False


class StreamConsumer:
    """Consumes sensor data messages from an async queue.

    Simulates a Kafka consumer, collecting messages and
    optionally triggering callbacks for processing.
    """

    def __init__(
        self,
        queue: asyncio.Queue[StreamMessage],
        batch_size: int = 10,
    ) -> None:
        self._queue = queue
        self._batch_size = batch_size
        self._running = False
        self._processed_count = 0
        self._callbacks: list[Any] = []

    def register_callback(self, callback: Any) -> None:
        """Register a callback to be invoked on each batch."""
        self._callbacks.append(callback)

    async def consume(self, max_messages: int | None = None) -> list[StreamMessage]:
        """Consume messages from the queue.

        Args:
            max_messages: Stop after this many messages. None = run until stopped.

        Returns:
            List of all consumed messages.
        """
        self._running = True
        all_messages: list[StreamMessage] = []
        batch: list[StreamMessage] = []

        logger.info("Starting stream consumer")

        while self._running:
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                batch.append(message)
                all_messages.append(message)
                self._processed_count += 1

                if len(batch) >= self._batch_size:
                    await self._process_batch(batch)
                    batch = []

                if max_messages and self._processed_count >= max_messages:
                    break

            except TimeoutError:
                if batch:
                    await self._process_batch(batch)
                    batch = []
                if self._queue.empty():
                    break

        # Process any remaining messages
        if batch:
            await self._process_batch(batch)

        logger.info("Consumer processed {} messages", self._processed_count)
        return all_messages

    async def _process_batch(self, batch: list[StreamMessage]) -> None:
        """Process a batch of messages through registered callbacks."""
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(batch)
            else:
                callback(batch)

    def stop(self) -> None:
        self._running = False


class StreamSimulator:
    """End-to-end streaming simulation orchestrator.

    Creates producer and consumer, manages the async event loop,
    and provides a simple interface for testing streaming scenarios.
    """

    def __init__(
        self,
        records_per_second: float = 10.0,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        self._queue: asyncio.Queue[StreamMessage] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.producer = StreamProducer(self._queue, records_per_second)
        self.consumer = StreamConsumer(self._queue, batch_size)

    async def run(
        self,
        df: pd.DataFrame,
        max_messages: int | None = None,
    ) -> list[StreamMessage]:
        """Run the streaming simulation.

        Args:
            df: Source DataFrame to stream.
            max_messages: Maximum messages to consume.

        Returns:
            List of consumed messages.
        """
        logger.info("Starting streaming simulation")

        producer_task = asyncio.create_task(
            self.producer.produce_from_dataframe(df)
        )
        messages = await self.consumer.consume(max_messages)

        self.producer.stop()
        await producer_task

        return messages

    def stop(self) -> None:
        """Stop both producer and consumer."""
        self.producer.stop()
        self.consumer.stop()
