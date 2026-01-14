# features = {
#     'rsi': 62,
#     'macd': 1.5,
#     'macd_signal': 1.2,
#     'bollinger_width': 0.04,
#     'news_sentiment': 0.6,
#     'social_volume': 8000,
#     'social_sentiment': 0.7,
#     'pe_ratio': 22,
#     'eps': 3.2,
#     'roe': 0.15,
#     'inflation_rate': 0.032,
#     'vwap': 98,
#     'price': 100,
#     'volume_change': 0.12,
#     'insider_activity': 0.03,
#     'short_ratio': 0.18,
#     'analyst_rating_change': 0.4,
#     'crude_oil_price': 84,
#     'treasury_yield_10yr': 0.045,
# }

# technical_weights = {
#     'rsi': 0.08,
#     'macd': 0.07,
#     'macd_signal': 0.05,
#     'sma_50': 0.04,
#     'ema_12': 0.04,
#     'bollinger_band_width': 0.03,
#     'bollinger_%b': 0.03,
#     'adx': 0.05,
#     'atr': 0.04,
#     'cci': 0.03,
#     'mfi': 0.03,
#     'obv': 0.04,
#     'stochastic_k': 0.03,
#     'stochastic_d': 0.03,
#     'ichimoku_cloud': 0.04,
#     'momentum': 0.05,
#     'roc': 0.03,
#     'golden_cross_flag': 0.03,
#     'death_cross_flag': -0.03,
#     'vwap': 0.04,
#     'volume_spike': 0.03,
# }

# ### 📉 2. Fundamental Metrics
# fundamental_weights = {
#     'eps_ttm': 0.07,
#     'pe_ratio': -0.05,
#     'peg_ratio': -0.04,
#     'pb_ratio': -0.02,
#     'ps_ratio': -0.01,
#     'ev_ebitda': -0.03,
#     'revenue_growth_yoy': 0.08,
#     'net_margin': 0.05,
#     'roe': 0.04,
#     'roa': 0.03,
#     'dividend_yield': 0.03,
#     'free_cash_flow_yield': 0.04,
#     'buyback_yield': 0.04,
#     'debt_to_equity': -0.04,
#     'interest_coverage': 0.03,
#     'institutional_holdings': 0.03,
#     'insider_holdings': 0.02
# }

# ### 🌍 3. Macroeconomic Indicators
# macro_weights = {
#     'cpi': -0.04,
#     'core_inflation': -0.03,
#     'unemployment_rate': -0.05,
#     'gdp_growth': 0.08,
#     'fed_rate': -0.07,
#     'real_interest_rate': -0.04,
#     'yield_curve_spread': 0.05,
#     'dxy': -0.03,
#     'oil_price': -0.02,
#     'gold_price': -0.01,
#     'vix': -0.05,
#     'retail_sales_growth': 0.04,
#     'manufacturing_pmi': 0.05,
#     'trade_balance': 0.02,
#     'central_bank_signal_flag': 0.05
# }


# ### 📊 4. Market Microstructure & Flow
# microstructure_weights = {
#     'short_ratio': 0.03,
#     'analyst': 0.05,
#     'treasury': 0.03,
#     'volume_change_pct': 0.05,
#     'vwap_deviation': 0.05,
#     'ask_bid_spread': -0.04,
#     'short_interest_pct': -0.03,
#     'put_call_ratio': -0.03,
#     'dark_pool_pct': 0.02,
#     'gamma_exposure': 0.04,
#     'delta_exposure': 0.03,
#     'options_open_interest_change': 0.03,
#     'iv_rank': -0.02,
#     'etf_rebalancing_flag': 0.03,
#     'block_trade_activity': 0.03
# }

# ### 🗞️ 5. Sentiment & News Data
# sentiment_weights = {
#     'news_sentiment_score': 0.06,
#     'twitter_sentiment': 0.04,
#     'reddit_sentiment': 0.04,
#     'analyst_rating_change': 0.05,
#     'target_price_shift': 0.05,
#     'ceo_sentiment_score': 0.04,
#     'insider_activity_score': 0.03,
#     'lawsuit_flag': -0.03,
#     'retail_sentiment_score': 0.03
# }

# ### 🧬 6. Stock-Specific Events
# event_weights = {
#     'earnings_surprise_pct': 0.08,
#     'guidance_shift': 0.06,
#     'product_launch_flag': 0.05,
#     'merger_announcement_score': 0.05,
#     'litigation_risk_score': -0.04,
#     'buyback_announcement_flag': 0.04,
#     'split_flag': 0.02,
#     'management_change_flag': -0.03,
#     'dividend_declaration_flag': 0.03,
#     'fda_approval_flag': 0.05
# }

# ### 🛰️ 7. Geopolitical & External Risks

# geopolitical_weights = {
#     'war_tension_index': -0.06,
#     'cyberattack_index': -0.04,
#     'sanctions_flag': -0.04,
#     'china_risk_score': -0.03,
#     'tail_risk_score': -0.05,
#     'global_supply_chain_score': -0.03,
#     'pandemic_flag': -0.05,
#     'natural_disaster_impact': -0.04
# }

# ### 🔗 Combined Weight Dictionary

# weights = {
#     **technical_weights,
#     **fundamental_weights,
#     **macro_weights,
#     **microstructure_weights,
#     **sentiment_weights,
#     **event_weights,
#     **geopolitical_weights
# }