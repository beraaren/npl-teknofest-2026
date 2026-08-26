import asyncio
import redis.asyncio as redis
import json
import logging

#: Tüketici döngüsü hata aldığında beklenecek süre (saniye). Bu bekleme
#: olmadan, Redis erişilemez olduğunda `while True` döngüsü hatayı anında
#: tekrar alıp sonsuz hızda dönerek CPU'yu %100'e çıkarır.
CONSUME_ERROR_BACKOFF_SEC = 1.0

logger = logging.getLogger(__name__)

async def get_redis_client(url: str = "redis://localhost:6379"):
    """Returns an async Redis client."""
    return redis.from_url(url, decode_responses=True)

async def publish_message(client: redis.Redis, stream_name: str, pydantic_model):
    """Publishes a Pydantic model to a Redis stream."""
    try:
        data = pydantic_model.model_dump_json()
        await client.xadd(stream_name, {"payload": data})
        logger.debug(f"Published to {stream_name}: {data[:100]}...")
    except Exception as e:
        logger.error(f"Error publishing to {stream_name}: {e}")

async def publish_raw(client: redis.Redis, stream_name: str, data: dict):
    """Ham dict'i bir Redis stream'ine yayınlar."""
    try:
        payload = json.dumps(data, ensure_ascii=False)
        await client.xadd(stream_name, {"payload": payload})
    except Exception as e:
        logger.error(f"Error publishing raw to {stream_name}: {e}")

async def consume_stream(
    client: redis.Redis,
    stream_name: str,
    group_name: str,
    consumer_name: str,
    block: int = 1000,
):
    """
    Generator that consumes messages from a Redis stream using a consumer group.
    Yields (message_id, payload_dict).
    On JSON parse error writes the raw message to <stream>.dlq and skips it.
    """
    try:
        await client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    while True:
        try:
            messages = await client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=1,
                block=block
            )
            for stream, msgs in messages:
                for msg_id, msg_data in msgs:
                    raw_payload = msg_data.get("payload", "")
                    try:
                        yield msg_id, json.loads(raw_payload)
                    except (json.JSONDecodeError, ValueError) as parse_err:
                        logger.error(
                            f"[DLQ] JSON parse error in {stream_name} msg {msg_id}: {parse_err}"
                        )
                        # Bozuk mesajı DLQ stream'ine yaz, ana akıştan ACK'le
                        dlq_stream = f"{stream_name}.dlq"
                        try:
                            await client.xadd(dlq_stream, {
                                "original_stream": stream_name,
                                "msg_id": msg_id,
                                "raw_payload": raw_payload,
                                "error": str(parse_err),
                            })
                        except Exception as dlq_err:
                            logger.error(f"DLQ write error: {dlq_err}")
                        # Bozuk mesajı ACK'le (pending'de bırakma)
                        await ack_message(client, stream_name, group_name, msg_id)
        except asyncio.CancelledError:
            # Servis kapanışında görev iptal edilir; bunu hata sayıp yeniden
            # denemek kapanışı engeller.
            raise
        except Exception as e:
            logger.error(f"Error consuming from {stream_name}: {e}")
            # Geri çekilme (backoff): Redis düştüğünde busy-loop'u önler.
            await asyncio.sleep(CONSUME_ERROR_BACKOFF_SEC)

async def ack_message(client: redis.Redis, stream_name: str, group_name: str, message_id: str):
    """Acknowledges a processed message."""
    await client.xack(stream_name, group_name, message_id)
