# db/models.py
from __future__ import annotations
from typing import List
from datetime import datetime
from sqlalchemy import String, Integer, ForeignKey, BLOB, DateTime, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("(datetime('now','localtime'))"))

    embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    # ArcFace 임베딩 float32 -> bytes 로 저장 (len = 4 * dim)
    dim: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(BLOB)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("(datetime('now','localtime'))"))

    user: Mapped[User] = relationship(back_populates="embeddings")
