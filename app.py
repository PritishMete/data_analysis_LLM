from fastapi import FastAPI, UploadFile, File, HTTPException
from analyzer.analyze import analyze_file
from privacy_policy import LOCAL_ONLY
import os
import shutil
import tempfile

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Data Analysis API Running", "privacy_mode": "local_only" if LOCAL_ONLY else "remote_allowed"}

@app.post("/analyze")
async def analyze(uploaded_file: UploadFile = File(...)):
    if LOCAL_ONLY:
        raise HTTPException(status_code=403, detail="Dataset upload disabled in local-only privacy mode.")
    suffix = os.path.splitext(uploaded_file.filename or "dataset")[1]
    fd, file_location = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        return analyze_file(file_location)
    finally:
        try:
            os.remove(file_location)
        except OSError:
            pass
