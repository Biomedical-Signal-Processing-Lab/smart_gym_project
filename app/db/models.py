# db/models.py
from __future__ import annotations
from typing import List
from datetime import datetime
from sqlalchemy import (
    String, Integer, ForeignKey, BLOB, DateTime, text, Float, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("(datetime('now','localtime'))")
    )

    embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    sessions: Mapped[List["WorkoutSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(BLOB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("(datetime('now','localtime'))")
    )

    user: Mapped[User] = relationship(back_populates="embeddings")

class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # 운동 종류: "squat", "lunge", "pushup", "plank"
    exercise: Mapped[str] = mapped_column(String(20), index=True)

    reps: Mapped[int] = mapped_column(Integer, default=0)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0)       
    duration_sec: Mapped[int] = mapped_column(Integer, default=0)

    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("(datetime('now','localtime'))")
    )
    ended_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("(datetime('now','localtime'))")
    )

    user: Mapped[User] = relationship(back_populates="sessions")

Index("ix_sessions_user_started", WorkoutSession.user_id, WorkoutSession.started_at)
