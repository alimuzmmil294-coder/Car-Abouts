"""
Car-Identifier & Vehicle Fleet Tracker
A Streamlit web application utilizing SQLAlchemy (SQLite) for persistence
and Google Gemini API (google-genai SDK) for multimodal car identification
and fleet analytics.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Import the official Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


# ==============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fleet_tracker.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_IMAGE_DIMENSION = 1024

CURRENT_YEAR = datetime.now().year
BODY_STYLES = ["Sedan", "SUV", "Truck", "Hatchback", "Van", "Coupe", "Wagon", "Convertible", "Other"]
VEHICLE_STATUSES = ["Active", "In Maintenance", "Inactive", "Sold"]


# ==============================================================================
# 2. DATABASE PERSISTENCE LAYER (SQLAlchemy + SQLite)
# ==============================================================================

class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    license_plate = Column(String(32), unique=True, nullable=False, index=True)
    make = Column(String(64), nullable=False)
    model = Column(String(64), nullable=False)
    year = Column(Integer, nullable=False)
    body_style = Column(String(32), nullable=False)
    color = Column(String(32), nullable=False)
    mileage = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="Active")
    assigned_driver = Column(String(128), nullable=True)
    notes = Column(Text, nullable=True)
    identification_source = Column(String(32), default="Manual")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "license_plate": self.license_plate,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "body_style": self.body_style,
            "color": self.color,
            "mileage": self.mileage,
            "status": self.status,
            "assigned_driver": self.assigned_driver or "Unassigned",
            "notes": self.notes or "",
            "identification_source": self.identification_source,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M") if self.updated_at else "",
        }


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Initialize database tables idempotently."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session scope."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ==============================================================================
# 3. REPOSITORY & VALIDATION LAYER
# ==============================================================================

def validate_vehicle_input(
    license_plate: str,
    make: str,
    model: str,
    year: int,
    body_style: str,
    color: str,
    mileage: float,
    status: str,
) -> Tuple[bool, Optional[str]]:
    """Validate vehicle attributes prior to persistence."""
    cleaned_plate = license_plate.strip().upper()
    if not cleaned_plate:
        return False, "License plate cannot be blank."
    if len(cleaned_plate) > 32:
        return False, "License plate exceeds maximum allowed length (32 characters)."

    if not make.strip():
        return False, "Vehicle Make is required."
    if not model.strip():
        return False, "Vehicle Model is required."

    if year < 1900 or year > (CURRENT_YEAR + 1):
        return False, f"Year must be between 1900 and {CURRENT_YEAR + 1}."

    if body_style not in BODY_STYLES:
        return False, f"Invalid body style: {body_style}."

    if not color.strip():
        return False, "Vehicle color is required."

    if mileage < 0.0:
        return False, "Odometer mileage must be greater than or equal to 0."
