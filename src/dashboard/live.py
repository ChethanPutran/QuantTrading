

from fastapi import FastAPI
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except Exception:
    SCHEDULER_AVAILABLE = False

from .integration import start_system_in_thread, get_stats, process_tick

app = FastAPI()

if SCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()

    def update_prediction():
        # This job can trigger the prediction pipeline periodically
        stats = get_stats()
        print("Prediction heartbeat - system stats keys:", list(stats.keys()))

    scheduler.add_job(update_prediction, 'interval', minutes=10)
    scheduler.start()
else:
    def update_prediction():
        pass


@app.on_event("startup")
def startup_event():
    # Ensure the trading system is running for live endpoints
    start_system_in_thread()


@app.get("/")
def read_root():
    return {"message": "Stock prediction API running."}


@app.get("/stats")
def read_stats():
    return get_stats()


@app.post("/tick/{price}")
def post_tick(price: float):
    process_tick(price)
    return {"accepted": True}
