from fastapi import FastAPI, Depends
from pydantic import BaseModel
from saferoute.data_access.repositories import SessionRepo, SessionLocal

# Initialize the FastAPI application
app = FastAPI(title="SafeRoute API", version="0.1.0")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic schema for incoming request validation
class SessionCreate(BaseModel):
    model_version: str

@app.get("/health")
def health_check():
    """Basic health check to ensure the API is running."""
    return {"status": "ok", "message": "SafeRoute backend is online."}

@app.post("/sessions")
def create_session(session_data: SessionCreate, db = Depends(get_db)):
    """Creates a new driving session in the SQLite database."""
    repo = SessionRepo(db)
    new_session = repo.create_session(model_version=session_data.model_version)
    
    return {
        "session_id": new_session.id, 
        "model_version": new_session.model_version,
        "status": "Session successfully created in WAL mode."
    }