import asyncio

from smart.trading_system.comm.event_bus import init_event_bus
from smart.trading_system.comm.events import TickEvent


def test_event_bus_publish_and_wait():
    bus = init_event_bus()

    received = []

    async def handler(ev):
        received.append(ev.price)

    bus.subscribe(TickEvent, handler)

    async def run():
        await bus.publish_and_wait(TickEvent(price=123.4, timestamp=0.0))

    asyncio.run(run())

    assert received == [123.4]
