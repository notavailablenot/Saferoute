"""Repository classes: the only place the Logic Tier touches the database."""
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

################-updated
from .models import Alert, Base, EvalRun, ModelVersion, PerfMetric, Session

DB_URL = os.getenv("SAFEROUTE_DB_URL", "sqlite:///saferoute.db")

def make_engine(url: str = DB_URL):
    engine = create_engine(url, connect_args={"check_same_thread": False})
################-Old
#Create SQLite Engine
#engine = create_engine("sqlite:///saferoute.db", connect_args={"check_same_thread": False})
#################
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    return engine


def init_db(url: str = DB_URL):
    """Create tables explicitly at app startup (not at import time)."""
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class SessionRepo:
    def __init__(self, db):
        self.db = db

    def create(self, model_version: str) -> Session:
        s = Session(model_version=model_version)
        self.db.add(s)
        self.db.commit()
        return s

    def close(self, session_id: int) -> None:
        s = self.db.get(Session, session_id)
        if s is not None:
            s.end_time = datetime.now(timezone.utc)
            self.db.commit()

    def save_telemetry(self, rows: list[PerfMetric]) -> None:
        self.db.add_all(rows)
        self.db.commit()


class AlertRepo:
    def __init__(self, db):
        self.db = db

    def save(self, alert: Alert) -> Alert:
        self.db.add(alert)
        self.db.commit()
        return alert

    def list(self, session_id: int) -> list[Alert]:
        return list(self.db.scalars(select(Alert).where(Alert.session_id == session_id)))


class EvalRepo:
    def __init__(self, db):
        self.db = db

    def save(self, run: EvalRun) -> EvalRun:
        self.db.add(run)
        self.db.commit()
        return run

    def get(self, run_id: int) -> EvalRun | None:
        return self.db.get(EvalRun, run_id)


class ModelRegistry:
    def __init__(self, db):
        self.db = db

    def register(self, model: ModelVersion) -> ModelVersion:
        self.db.add(model)
        self.db.commit()
        return model

    def get_approved(self) -> ModelVersion | None:
        stmt = (select(ModelVersion).where(ModelVersion.approved.is_(True))
                .order_by(ModelVersion.created_at.desc()))
        return self.db.scalars(stmt).first()

    def set_status(self, model_id: int, approved: bool) -> None:
        m = self.db.get(ModelVersion, model_id)
        if m is None:
            raise KeyError(f"model {model_id} not registered")
        m.approved = approved
        self.db.commit()
