from saferoute.data_access.models import Alert, ModelVersion, PerfMetric
from saferoute.data_access.repositories import AlertRepo, ModelRegistry, SessionRepo, init_db


def test_session_alert_telemetry_roundtrip(tmp_path):
    Factory = init_db(f"sqlite:///{tmp_path/'t.db'}")
    with Factory() as db:
        s = SessionRepo(db).create("yolo11n-ph-v1")
        AlertRepo(db).save(Alert(session_id=s.id, track_id=12, sign_class="STOP", tier=1, confidence=0.94))
        SessionRepo(db).save_telemetry([PerfMetric(session_id=s.id, fps=27.4, inference_ms=18.0, end_to_end_ms=74.0)])
        SessionRepo(db).close(s.id)
        assert [a.sign_class for a in AlertRepo(db).list(s.id)] == ["STOP"]
        assert s.end_time is not None


def test_only_approved_model_is_returned(tmp_path):
    Factory = init_db(f"sqlite:///{tmp_path/'t.db'}")
    with Factory() as db:
        reg = ModelRegistry(db)
        m = reg.register(ModelVersion(name="v1", path="models/v1.onnx", sha256="abc"))
        assert reg.get_approved() is None
        reg.set_status(m.id, True)
        assert reg.get_approved().name == "v1"
