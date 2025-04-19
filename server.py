import dotenv
dotenv.load_dotenv()

from utilities import ai, utils

device = ai.optimize_memory()

print(f"Using: {device}")

import fastapi
import sqlite3
import pandas
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
import pandas

conn = sqlite3.connect('db.db')

app = fastapi.FastAPI()

class SignupData(BaseModel):
    username: str
    password: str

# Serve index.html at root
@app.get("/")
async def read_root():
    return "<h1> T5 Chat </h1>This is mainly used for API endpoints, not as a website!"

@app.post('/api/signup')
async def signup(username, password):
    user = utils.Users(0, username, password)
    #username = data.username
    #password = data.password
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, display_name TEXT, description TEXT, instructions TEXT)")
    if not conn.execute(f"SELECT * FROM users WHERE username = '{username}'").fetchone():
        user.save_sql(conn)
        return {"success": True, "message": "User created"}
    else:
        return {"success": False, "message": "Username already exists"}

@app.post('/api/login')
async def login(data: SignupData):
    username = data.username
    password = data.password

    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, display_name TEXT, description TEXT, instructions TEXT)")
    user = utils.Users(0, username, password)
    results = conn.execute(f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'").fetchone()
    if results:
        return {"success": True, "message": "User logged in"}
    else:
        return {"success": False, "message": "Invalid username or password"}

@app.get('/api/convs')
def convs(data: SignupData):
    username = data.username
    password = data.password
    # Verify user credentials first
    user_id = conn.execute(f"SELECT id FROM users WHERE username = '{username}' AND password = '{password}'").fetchone()
    
    if not user_id:
        return {"success": False, "message": "Invalid username or password"}
    
    conn.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, messages TEXT)")
    results = conn.execute(f"SELECT id, title FROM conversations WHERE user_id = '{user_id[0]}'").fetchall()
    if results:
        # Convert to list of dictionaries for better JSON serialization
        conversations = []
        for row in results:
            conversations.append({"id": row[0], "title": row[1]})
        return {"success": True, "conversations": conversations}
    else:
        return {"success": True, "conversations": []}

@app.get('/api/convs/{conversation_id}')
def get_conversation(conversation_id: int, username: str, password: str):
    # Get user ID from username
    user_id = conn.execute(f"SELECT id FROM users WHERE username = '{username}' AND password = '{password}'").fetchone()
    
    if not user_id:
        return {"success": False, "message": "User not found"}
    
    # Get the conversation
    conversation = conn.execute(
        "SELECT id, title, messages FROM conversations WHERE id = ? AND user_id = ?", 
        (conversation_id, user_id[0])
    ).fetchone()
    
    if not conversation:
        return {"success": False, "message": "Conversation not found"}
    
    # Parse messages JSON
    messages = json.loads(conversation[2])
    
    # Return conversation with parsed messages
    return {
        "success": True,
        "conversation": {
            "id": conversation[0],
            "title": conversation[1],
            "messages": messages
        }
    }

@app.post('/api/convs/create')
def create_conv(username: str, password: str, prompt: str):
    # Get user ID from username and verify password
    user_id = conn.execute(f"SELECT id FROM users WHERE username = '{username}' AND password = '{password}'").fetchone()
    
    if not user_id:
        return {"success": False, "message": "Invalid username or password"}
    
    # Create initial message structure
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    intructions = conn.execute(f"SELECT instructions FROM users WHERE id = '{user_id[0]}'").fetchone()
    if instructions:
        messages.insert(0, {
            "role": "system",
            "content": intructions[0]
        })

    # Convert messages list to JSON string
    messages_json = json.dumps(messages)
    
    # Create a title from the prompt (first 30 chars)
    title = prompt[:30] + "..." if len(prompt) > 30 else prompt
    
    # Insert new conversation
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (user_id, title, messages) VALUES (?, ?, ?)",
        (user_id[0], title, messages_json)
    )
    conn.commit()
    
    # Return the new conversation ID
    return {"success": True, "conversation_id": cursor.lastrowid}

@app.post('/api/convs/add_message')
def add_message(conversation_id: int, username: str, password: str, message: str, model: str):
    # Get user ID from username and verify password
    user_id = conn.execute(f"SELECT id FROM users WHERE username = '{username}' AND password = '{password}'").fetchone()
    
    if not user_id:
        return {"success": False, "message": "Invalid username or password"}
    
    # Get the conversation
    conversation = conn.execute(
        "SELECT messages FROM conversations WHERE id = ? AND user_id = ?", 
        (conversation_id, user_id[0])
    ).fetchone()
    
    if not conversation:
        return {"success": False, "message": "Conversation not found"}
    
    # Parse existing messages
    messages = json.loads(conversation[0])
    
    # Add new message
    messages.append({
        "role": 'user',
        "content": message
    })

    m = {'llama3.2': ai.Available.llamai_3_2, 'llama3.2think': ai.Available.llamai_3_2think, 'llama3.2think2': ai.Available.llamai_3_2think2, 'llama3.2think3': ai.Available.llamai_3_2think3}

    ai_model = m[model]

    response = ai.produce_output(ai_model, messages)

    messages.append({
        "role": 'assistant',
        "content": response  
    })
    
    # Convert messages list back to JSON string
    messages_json = json.dumps(messages)
    
    # Update the conversation
    conn.execute(
        "UPDATE conversations SET messages = ? WHERE id = ?",
        (messages_json, conversation_id)
    )
    conn.commit()
    
    return {"success": True, "message": "Message added"}

if __name__ == "__main__":
    raise RuntimeError("Please run this with `fastapi run server.py`")
