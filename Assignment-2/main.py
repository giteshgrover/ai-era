from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

# Handle file upload
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("static/uploads", exist_ok=True)
        
        # Save the file
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Return file details
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get available animals
@app.get("/animals")
async def get_animals():
    return {
        "animals": [
            {"name": "cat", "image": "/static/images/cat.jpg"},
            {"name": "dog", "image": "/static/images/dog.jpg"},
            {"name": "elephant", "image": "/static/images/elephant.jpg"}
        ]
    }