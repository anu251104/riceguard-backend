import os
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

NVIDIA_API_KEY = os.getenv("API_KEY")

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)




app = FastAPI()

users = {}
contact_messages = []

DISEASE_TREATMENTS = {

    "Tungro": {
        "chemical": [
            "Spray Imidacloprid 17.8% SL @ 0.3 ml per liter of water",
            "Apply Thiamethoxam 25% WG @ 0.25 g per liter",
            "Control green leafhopper using Monocrotophos 36% SL @ 1.6 ml per liter"
        ],
        "organic": [
            "Spray Neem oil 3 ml per liter every 7 days",
            "Use yellow sticky traps to control leafhoppers",
            "Remove and destroy infected plants immediately",
            "Use resistant rice varieties"
        ]
    },

    "Sheath Blight": {
        "chemical": [
            "Spray Validamycin 3% L @ 2 ml per liter",
            "Apply Hexaconazole 5% EC @ 1 ml per liter",
            "Use Propiconazole 25% EC @ 1 ml per liter"
        ],
        "organic": [
            "Apply Trichoderma harzianum in soil",
            "Avoid excessive nitrogen fertilizer",
            "Maintain proper plant spacing for air circulation",
            "Improve field drainage"
        ]
    },

    "Leaf Scald": {
        "chemical": [
            "Spray Mancozeb 75% WP @ 2 g per liter",
            "Apply Copper oxychloride @ 2.5 g per liter"
        ],
        "organic": [
            "Use certified disease-free seeds",
            "Remove infected leaves",
            "Apply compost-enriched soil to improve plant immunity",
            "Maintain proper irrigation management"
        ]
    },

    "Brown Spot": {
        "chemical": [
            "Spray Carbendazim 1 g per liter",
            "Apply Mancozeb 2 g per liter",
            "Use Propiconazole 1 ml per liter"
        ],
        "organic": [
            "Apply neem cake to soil",
            "Use Pseudomonas fluorescens as bio-control agent",
            "Avoid water stress",
            "Ensure balanced fertilization (especially potassium)"
        ]
    },

    "Hispa": {
        "chemical": [
            "Spray Chlorpyrifos 20% EC @ 2 ml per liter",
            "Apply Cypermethrin 25% EC @ 1 ml per liter",
            "Use Quinalphos 25% EC @ 2 ml per liter"
        ],
        "organic": [
            "Handpick and destroy adult beetles",
            "Install light traps",
            "Spray neem seed kernel extract (NSKE) 5%",
            "Encourage natural predators like spiders"
        ]
    },

    "Leaf Blast": {
        "chemical": [
            "Spray Tricyclazole 75% WP @ 0.6 g per liter",
            "Apply Isoprothiolane @ 1.5 ml per liter",
            "Use Carbendazim 1 g per liter"
        ],
        "organic": [
            "Apply Pseudomonas fluorescens",
            "Avoid excess nitrogen fertilizer",
            "Use resistant varieties",
            "Ensure proper drainage"
        ]
    },

    "Healthy": {
        "chemical": [],
        "organic": [
            "No treatment required",
            "Maintain proper irrigation schedule",
            "Use balanced fertilizers",
            "Monitor crop regularly for early detection"
        ]
    }
}


# Allow frontend (React) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






# 🔥 LOAD YOUR RETRAINED MODEL HERE
MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

model.to("cpu")








def hash_password(password: str):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

@app.get("/")
def root():
    return {"status": "RiceGuard backend running 🚀"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print("📥 Received file:", file.filename)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    probs = results[0].probs
    top1 = probs.top1
    confidence = float(probs.top1conf)
    class_name = results[0].names[top1]

    

    disease_name = class_name

    treatment = DISEASE_TREATMENTS.get(disease_name, {
        "chemical": ["No data available"],
        "organic": ["No data available"]
    })

    return {
        "prediction": disease_name,
        "confidence": round(confidence * 100, 2),
        "treatment": treatment
    }


#         }

@app.get("/")
def root():
    return {"status": "Backend working"}

    
@app.post("/contact")
def save_contact(data: dict = Body(...)):
    message = {
        "firstName": data.get("firstName"),
        "lastName": data.get("lastName"),
        "email": data.get("email"),
        "subject": data.get("subject"),
        "message": data.get("message")
    }

    contact_messages.append(message)

    return {"status": "Message saved successfully"}




@app.get("/admin/messages")
def get_messages():
    return contact_messages




@app.post("/signup")
def signup(data: dict = Body(...)):
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="All fields are required")

    if email in users:
        raise HTTPException(status_code=400, detail="Account already exists")

    users[email] = {
        "name": name,
        "password": hash_password(password)
    }

    return {"message": "Signup successful"}


@app.post("/login")
def login(data: dict = Body(...)):
    email = data.get("email")
    password = hash_password(data.get("password"))

    if email not in users:
        raise HTTPException(status_code=404, detail="Account not found")

    if users[email]["password"] != password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    return {
        "token": "demo-token",
        "user": users[email]["name"]
    }

@app.post("/chat")
async def chat(data: dict = Body(...)):
    user_message = data.get("message")
    history = data.get("history", [])

    if not user_message:
        raise HTTPException(status_code=400, detail="Message required")

    # System role (agriculture expert)
    messages = [
       {
    "role": "system",
    "content": """You are RiceGuard AI, a helpful and intelligent agricultural assistant specializing in rice diseases.

Rules:
- Answer only what the user asks.
- Keep responses clear and concise (3–6 sentences).
- Do NOT give long essays.
- Do NOT always structure answers into causes/treatment unless the user asks.
- Be conversational and natural like ChatGPT.
- If the question is not related to rice or agriculture, politely say you specialize in crop health.
"""
}

    ]

    # Add previous conversation history
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add latest user message
    messages.append({
        "role": "user",
        "content": user_message
    })

    try:
        completion = client.chat.completions.create(
            model="meta/llama3-8b-instruct",
            messages=messages,
            temperature=0.6,
            max_tokens=300,
            top_p=0.9,

        )

        reply = completion.choices[0].message.content

        return {"reply": reply}

    except Exception as e:
        print("NVIDIA ERROR:", str(e))
        raise HTTPException(status_code=500, detail="AI service failed")




