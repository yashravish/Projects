# websocket.py
@app.websocket("/ws/{ticket_id}")
async def websocket_endpoint(websocket: WebSocket, ticket_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast message to all connected clients for this ticket
            await manager.broadcast(f"Ticket {ticket_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)