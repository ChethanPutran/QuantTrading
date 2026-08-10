from execution.simulator import TradingSimulator


def test_simulator_buy_sell_and_reset():
    sim = TradingSimulator(initial_balance=1000.0, initial_price=10.0, transaction_cost=0.001, slippage=0.0)
    # buy
    res = sim.step(1, current_price=10.0, order_size=0.5)
    assert res['shares_traded'] >= 0
    # update price and sell
    sim.update_price(12.0)
    res2 = sim.step(-1, current_price=12.0, order_size=1.0)
    assert 'new_position' in res2
    stats = sim.get_statistics()
    assert 'n_steps' in stats
    sim.reset()
    s = sim.get_state()
    assert s['position'] == 0.0
