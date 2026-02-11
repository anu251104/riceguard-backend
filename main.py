from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import hashlib


app = FastAPI()

users = {}
contact_messages = []


# Allow frontend (React) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 LOAD YOUR RETRAINED MODEL HERE
MODEL_PATH = "runs/clasify/runs/classify/weights/best.pt"

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

    return {
        "prediction": class_name,
        "confidence": round(confidence * 100, 2)
    }


@app.post("/chat")
async def chat(data: dict):
    message = data.get("message", "")

    # simple placeholder response (can upgrade later)
    return {
        "response": f"I detected rice plant disease-related query: '{message}'"
    }

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
