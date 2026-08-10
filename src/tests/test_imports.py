def test_core_imports():
    import app
    import dashboard.trading_view
    import dashboard.live
    import data.replay.loader

    assert hasattr(app, 'TradingSystem')
    assert hasattr(data.replay.loader, 'load_prices_from_csv')
