"""Task 5 - database trail. One row per prediction request.

Answers: what was decided (label, confidence, prob_ai, threshold), when
(created_at, UTC), by which model (model_name, model_version, inference_mode)
and how confidently (confidence).
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    prob_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    inference_mode: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float] = mapped_column(Float)


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def add(self, **fields) -> Prediction:
        with self.Session() as s:
            row = Prediction(**fields)
            s.add(row)
            s.commit()
            return row

    def recent(self, limit=100):
        with self.Session() as s:
            return list(s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))

    def get(self, pred_id):
        with self.Session() as s:
            return s.get(Prediction, pred_id)
