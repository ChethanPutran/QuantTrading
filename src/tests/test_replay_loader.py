import asyncio
import tempfile
import os

from data.replay.loader import load_prices_from_csv, replay_price_series


def test_load_prices_from_csv_and_replay():
    # create temporary CSV
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        with open(path, 'w') as f:
            f.write('Date,Close,Volume\n')
            f.write('2020-01-01,100.0,1000\n')
            f.write('2020-01-02,101.5,1100\n')
            f.write('2020-01-03,99.0,900\n')

        prices = load_prices_from_csv(path, price_column='Close')
        assert isinstance(prices, list)
        assert len(prices) == 3

        collected = []

        async def collect():
            async for tick in replay_price_series(prices, delay_per_tick=0):
                collected.append(tick)

        asyncio.run(collect())
        assert len(collected) == 3
        assert all(hasattr(t, 'price') for t in collected)
    finally:
        os.remove(path)
