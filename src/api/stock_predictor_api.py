from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from app.predict import run_prediction_for_all_tickers
from app.database import init_db

app = FastAPI()
scheduler = BackgroundScheduler()

# Schedule the prediction job every 10 minutes
scheduler.add_job(run_prediction_for_all_tickers, 'interval', minutes=10)
scheduler.start()

@app.on_event("startup")
def startup_event():
    init_db()
    run_prediction_for_all_tickers()

@app.get("/")
def root():
    return {"message": "Stock prediction API is running."}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    return run_prediction_for_all_tickers(ticker_override=ticker)

@app.get("/history/{ticker}")
def history(ticker: str):
    from app.database import get_prediction_history
    return get_prediction_history(ticker)
