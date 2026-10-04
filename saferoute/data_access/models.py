from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Session(Base):
    """Tracks each driving/monitoring session"""
    __tablename__ = "Sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=False)
    model_version = Column(String, nullable=False)

    #Relationships to link alerts and telemetry to this specific session
    alerts = relationship("Alert", back_populates="Session")
    telemetry = relationship("PerfMetric", back_populates="Session")

class Alert(Base):
    """Logs each unique traffic sign alert issued to the driver"""
    __tablename__ = "Alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    sign_class = Column(String, nullable=False)
    tier = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)

    class PerfMetrics(Base):
        """Logs per-sercond performance telemetry (FPS and Latency)"""
        __tablename__ = "Perf_Metrics"

        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(Integer, ForeignKey=("sessions.id"), nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow)
        fps = Column(Float, nullable=False)
        inference_ms = Column(Float, nullable=False)
        end_to_end = Column(Float, nullable=False)

        session = relationship("Session", back_populates="telemetry")