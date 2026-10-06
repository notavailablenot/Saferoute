from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from .models import Base, Session, Alert, PerfMetric

#Create SQLite Engine
engine = create_engine("sqlite:///saferoute.db", connect_args={"check_same_thread": False})

#Enable WAL mode and optimize synchronization on connection
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()

#Create Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

class SessionRepo:
    """Handles CRUD operations for diving sessions and telemetry"""
    def __init__(self, db):
        self.db = db

    def create_session(self, model_version: str) -> Session:
        new_session = Session(model_version=model_version)
        self.db.add(new_session)
        self.db.commit()
        self.db.refresh(new_session)
        return new_session

    def save_telemetry(self, telemetry_rows: list[PerfMetric]):
        self.db.add_all(telemetry_rows)
        self.db.commit()

class AlertRepo:
    """Handles CRUD operations for driver alerts"""
    def __init__(self, db):
        self.db = db

    def save_alert(self, alert: Alert):
        self.db.add(alert)
        self.db.commit()

    def list_by_session(self, session_id: int):
        return self.db.query(Alert).filter(Alert.session_id == session_id).all()

class EvalRepo:
    """Handles CRUD operations for model evaluation runs"""
    def __init__(self,db):
        self.db = db

    def save(self, run):
        self.db.add(run)
        self.db.commit()

    def get(self,  run_id: str):
        #Need EvalRun Model
        pass

class ModelRegistry:
    """Manages registered model versions and their approval status."""
    def __init__(self, db):
        self.db = db

    def register(self, model):
        self.db.add(model)
        self.db.commit()

    def get_approved(self):
        #Update
        pass

    def set_status(self, model_id: int, status: str):
        #Update
        pass