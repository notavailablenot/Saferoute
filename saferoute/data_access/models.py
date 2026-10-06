"""SQLAlchemy ORM models for the SafeRoute Data Access Tier (SQLite)."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Session(Base):
    ############################################-New
    """One driving/monitoring session (UC-01)."""
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # set on stop
    model_version: Mapped[str] = mapped_column(String, nullable=False)

    alerts: Mapped[list["Alert"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    telemetry: Mapped[list["PerfMetric"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Alert(Base):
    """One de-duplicated driver alert (one per confirmed track)."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    sign_class: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="alerts")


class PerfMetric(Base):
    """Per-second runtime telemetry (FR-05)."""
    __tablename__ = "perf_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    end_to_end_ms: Mapped[float] = mapped_column(Float, nullable=False)

    session: Mapped["Session"] = relationship(back_populates="telemetry")


class ModelVersion(Base):
    """Model registry entry (UC-02): only approved models may be loaded by UC-01."""
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)           # e.g. yolo11n-ph-v1
    path: Mapped[str] = mapped_column(String, nullable=False)           # models/yolo11n-ph-v1.onnx
    sha256: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    precision: Mapped[str] = mapped_column(String, default="FP32")      # FP32 / FP16 / INT8
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EvalRun(Base):
    """One evaluation run (UC-02), traceable to model hash and git commit."""
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), nullable=False)
    split: Mapped[str] = mapped_column(String, default="test")
    map50: Mapped[float | None] = mapped_column(Float)
    map50_95: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    git_commit: Mapped[str | None] = mapped_column(String)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
      
###################################################-Old
   #"""Tracks each driving/monitoring session"""
   #__tablename__ = "sessions"

   #id = Column(Integer, primary_key=True, autoincrement=True)
   #start_time = Column(DateTime, default=datetime.utcnow)
   #end_time = Column(DateTime, nullable=True)
   #model_version = Column(String, nullable=False)

    # Relationships to link alerts and telemetry to this specific session
   #alerts = relationship("Alert", back_populates="session")
   #telemetry = relationship("PerfMetric", back_populates="session")


#class Alert(Base):
   #"""Logs each unique traffic sign alert issued to the driver"""
   #__tablename__ = "alerts"

   #id = Column(Integer, primary_key=True, autoincrement=True)
   #session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
   #track_id = Column(Integer, nullable=False)
   #sign_class = Column(String, nullable=False)
   #tier = Column(Integer, nullable=False)
   #timestamp = Column(Float, nullable=False)
    
   #session = relationship("Session", back_populates="alerts")


#class PerfMetric(Base):
   #"""Logs per-second performance telemetry (FPS and Latency)"""
   #__tablename__ = "perf_metrics"

   #id = Column(Integer, primary_key=True, autoincrement=True)
   #session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
   #timestamp = Column(DateTime, default=datetime.utcnow)
   #fps = Column(Float, nullable=False)
   #inference_ms = Column(Float, nullable=False)
   #end_to_end_ms = Column(Float, nullable=False)
    
   #session = relationship("Session", back_populates="telemetry")


#class EvalRun(Base):
    #"""Stores model evaluation metrics."""
    #__tablename__ = "eval_runs"
    
    #id = Column(String, primary_key=True) 
    #model_hash = Column(String, nullable=False)
    #dataset_version = Column(String, nullable=False)
    #map50 = Column(Float)
    #map50_95 = Column(Float)
    #p95_latency = Column(Float)


#class RegisteredModel(Base):
    #"""Tracks model versions and approval status."""
    #__tablename__ = "model_registry"
    
    #id = Column(Integer, primary_key=True, autoincrement=True)
    #model_hash = Column(String, unique=True, nullable=False)
    #status = Column(String, default="Not Approved")
##################################################################
