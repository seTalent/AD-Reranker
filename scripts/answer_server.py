# answer_server.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from models.qwen import AnswerModel
import hydra
import uvicorn
from uvicorn import Config, Server

app = FastAPI()
answer_model = None

class CompletionInput(BaseModel):
    completions: list
    kwargs: dict = {}

@app.post("/reward")
async def reward_handler(data: CompletionInput):
    global answer_model
    try:
        rewards = answer_model.call(data.completions, data.kwargs)
        return {"rewards": rewards}
    except Exception as e:
        return {"error": str(e)}

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    global answer_model
    answer_model = AnswerModel(cfg.models.answer)
    print("[Server] AnswerModel initialized")

    # ✅ 启动 FastAPI 服务
    config = Config(app=app, host="0.0.0.0", port=8000, reload=False)
    server = Server(config)
    server.run()

if __name__ == "__main__":
    main()
