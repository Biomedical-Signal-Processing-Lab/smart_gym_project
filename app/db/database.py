# db/database.py
from __future__ import annotations
import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

def make_sqlite_url(db_path: str) -> str:
    db_path = os.path.abspath(db_path)
    return f"sqlite:///{db_path}"

def create_engine_and_session(db_path: str):
    url = make_sqlite_url(db_path)
    engine = create_engine(url, future=True)

    # SQLite 튜닝 (외래키/WAL/동기화)
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    return engine, SessionLocal

def init_db(engine):
    # 테이블 생성
    from db.models import Base  # 지연 임포트(순환 참조 방지)
    Base.metadata.create_all(engine)
