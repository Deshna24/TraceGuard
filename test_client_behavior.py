import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

task_finished = False

async def my_bg_task():
    global task_finished
    await asyncio.sleep(1)
    task_finished = True
    print("Background task finished!")

@app.post("/start")
async def start():
    asyncio.create_task(my_bg_task())
    return {"status": "started"}

def test_without_context():
    global task_finished
    task_finished = False
    client = TestClient(app)
    client.post("/start")
    import time
    time.sleep(1.5)
    print("Without context:", task_finished)

def test_with_context():
    global task_finished
    task_finished = False
    with TestClient(app) as client:
        client.post("/start")
        import time
        time.sleep(1.5)
    print("With context:", task_finished)

if __name__ == "__main__":
    test_without_context()
    test_with_context()
