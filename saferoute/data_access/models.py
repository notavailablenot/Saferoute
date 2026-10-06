from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Session(Base):
    """Tracks each driving/monitoring session"""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    model_version = Column(String, nullable=False)

    # Relationships to link alerts and telemetry to this specific session
    alerts = relationship("Alert", back_populates="session")
    telemetry = relationship("PerfMetric", back_populates="session")


class Alert(Base):
    """Logs each unique traffic sign alert issued to the driver"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    sign_class = Column(String, nullable=False)
    tier = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    
    session = relationship("Session", back_populates="alerts")


class PerfMetric(Base):
    """Logs per-second performance telemetry (FPS and Latency)"""
    __tablename__ = "perf_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    fps = Column(Float, nullable=False)
    inference_ms = Column(Float, nullable=False)
    end_to_end_ms = Column(Float, nullable=False)

    session = relationship("Session", back_populates="telemetry")


class EvalRun(Base):
    """Stores model evaluation metrics."""
    __tablename__ = "eval_runs"
    
    id = Column(String, primary_key=True) 
    model_hash = Column(String, nullable=False)
    dataset_version = Column(String, nullable=False)
    map50 = Column(Float)
    map50_95 = Column(Float)
    p95_latency = Column(Float)


class RegisteredModel(Base):
    """Tracks model versions and approval status."""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_hash = Column(String, unique=True, nullable=False)
    status = Column(String, default="Not Approved")