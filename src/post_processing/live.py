
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()
scheduler = BackgroundScheduler()

def update_prediction():
    print("Running prediction...")
    # Call full prediction pipeline here

scheduler.add_job(update_prediction, 'interval', minutes=10)
scheduler.start()

@app.get("/")
def read_root():
    return {"message": "Stock prediction API running."}
