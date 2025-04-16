from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List
import os
import uvicorn
import asyncio
from pydantic import BaseModel
import json
import tomllib

# my modules
from agent import Agent

# Set the FastAPI listen port as an environment variable or default to 8000
LISTEN_PORT = int(os.getenv("REST_API_PORT", 8000))
listen = os.environ.get("REST_API_LISTEN", "true").lower() == "true"

app = FastAPI(
    title="FastAPI REST API",
    description="A simple FastAPI REST API with WebSocket support",
)

if __name__ == "__main__":
    if listen:
        print(f"Starting FastAPI server on port {LISTEN_PORT}...")
        uvicorn.run("rest_api:app", host="0.0.0.0", port=LISTEN_PORT, reload=True)
    else:
        print(
            "FastAPI server is not set to listen. Set REST_API_LISTEN to 'true' to enable it."
        )


# In-memory storage for demonstration purposes


class UserProfile(BaseModel):
    id: str
    name: str
    # age: int
    # phone: str
    # sex: str
    # income: str
    # job: str
    # email: str
    # lifestyle: str
    # health_problems: str
    # health_goals: str
    # seeking_information: str


data_store = {"users": []}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Health Agent REST API"}


@app.get("/users")
async def get_users():
    return JSONResponse(
        content={"users": [user.dict() for user in data_store["users"]]}
    )


@app.post("/users")
async def add_item(users: List[UserProfile]):
    for user in users:
        data_store["users"].append(user)
    return JSONResponse(
        content={
            "message": "Users added successfully",
            "users": [user.id for user in users],
        }
    )


@app.delete("/users")
async def clear_users(users: List[UserProfile] = None):
    if users is None:
        return JSONResponse(
            content={"message": "No users specified, nothing was deleted."}
        )

    user_ids_to_remove = {user.id for user in users}
    data_store["users"] = [
        user for user in data_store["users"] if user.id not in user_ids_to_remove
    ]
    return JSONResponse(
        content={
            "message": "Specified users removed successfully",
            "removed_users": list(user_ids_to_remove),
        }
    )


###############################################################################
# WebSocket for streaming output
###############################################################################
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)


manager = ConnectionManager()


rag_app = Agent().get_app()


def stream_response(inputs, config):
    for output in rag_app.stream(inputs, config):
        for node_name, node_state in output.items():
            if not node_state:
                continue

            if node_state.get("thinking"):
                json_response = {
                    "thinking": node_state["thinking"],
                }
                yield json.dumps(json_response, ensure_ascii=False)

            if node_state.get("answer"):
                json_response = {
                    "answer": node_state["answer"],
                }
                yield json.dumps(json_response, ensure_ascii=False)


def run_conversation(user_input):
    thread_id = "rest_api_test"

    inputs = {"question": user_input}
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in stream_response(inputs, config):
        yield chunk


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for a message from the client
            user_input = await websocket.receive_text()
            print(f"Received message: {user_input}")

            for message in run_conversation(user_input):
                print(f"Sending message: {message}")
                await manager.broadcast(f"{message}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
