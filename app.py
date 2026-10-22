"""
Car Identifier & Inventory Tracker
Built with Streamlit, SQLite (embedded local persistence), and Google Gemini Flash.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import io
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import requests
import streamlit as st

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CarTracker")

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

DB_FILE_PATH: str = os.getenv("CAR_TRACKER_DB_PATH", "car_inventory.db")

DEFAULT_CURRENCY: str = (
    st.secrets.get("APP_DEFAULT_CURRENCY")
    if hasattr(st, "secrets") and "APP_DEFAULT_CURRENCY" in st.secrets
    else os.getenv("APP_DEFAULT_CURRENCY", "PKR")
)

ENABLE_AI_FEATURES: bool = str(
    st.secrets.get("ENABLE_AI_FEATURES")
    if hasattr(st, "secrets") and "ENABLE_AI_FEATURES" in st.secrets
    else os.getenv("ENABLE_AI_FEATURES", "true")
).lower() == "true"

GEMINI_API_KEY: Optional[str] = (
    st.secrets.get("GEMINI_API_KEY")
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets
    else os.getenv("GEMINI_API_KEY")
)

SUPPORTED_BODY_TYPES: List[str] = [
    "Sedan",
    "SUV",
    "Hatchback",
    "Crossover",
    "Coupe",
    "Convertible",
    "Truck / Pickup",
    "Van / Minivan",
    "Wagon",
    "Other",
]

FUEL_TYPES: List[str] = ["Petrol", "Diesel", "Hybrid", "Electric", "CNG / LPG", "Other"]

# ==============================================================================
# 2. SQLITE DATA PERSISTENCE LAYER
# ==============================================================================


class CarInventoryDatabase:
    """Encapsulates local, thread-safe SQLite operations for car inventory."""

    @staticmethod
    def get_connection() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def initialize_db(cls) -> None:
        """Initializes database schema and indexes."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cars (
                    id TEXT PRIMARY KEY,
                    brand TEXT NOT NULL,
                    model TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    body_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    fuel_type TEXT NOT NULL,
                    horsepower TEXT,
                    key_features TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_brand_model ON cars (brand, model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_price ON cars (price)")
            conn.commit()
            logger.info("SQLite car inventory schema verified.")

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cars ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def create(cls, data: Dict[str, Any]) -> str:
        import uuid

        record_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cars (
                    id, brand, model, year, body_type, price, currency,
                    fuel_type, horsepower, key_features, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    data["brand"],
                    data["model"],
                    int(data["year"]),
                    data["body_type"],
                    float(data["price"]),
                    data["currency"],
                    data["fuel_type"],
                    data.get("horsepower", ""),
                    data.get("key_features", ""),
                    data.get("notes", ""),
                    now,
                    now,
                ),
            )
            conn.commit()
            return record_id

    @classmethod
    def update(cls, record_id: str, data: Dict[str, Any]) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE cars
                SET brand = ?, model = ?, year = ?, body_type = ?, price = ?,
                    currency = ?, fuel_type = ?, horsepower = ?, key_features = ?,
                    notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    data["brand"],
                    data["model"],
                    int(data["year"]),
                    data["body_type"],
                    float(data["price"]),
                    data["currency"],
                    data["fuel_type"],
                    data.get("horsepower", ""),
                    data.get("key_features", ""),
                    data.get("notes", ""),
                    now,
                    record_id,
                ),
            )
            if cursor.rowcount == 0:
                raise FileNotFoundError(f"Vehicle record {record_id} not found.")
            conn.commit()

    @classmethod
    def delete(cls, record_id: str) -> None:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cars WHERE id = ?", (record_id,))
            if cursor.rowcount == 0:
                raise FileNotFoundError(f"Vehicle record {record_id} not found.")
            conn.commit()


# ==============================================================================
# 3. AI INTEGRATION LAYER (GOOGLE GEMINI FLASH)
# ==============================================================================


class GeminiCarAnalyst:
    """Handles vehicle identification and spec extraction via Gemini 1.5 Flash."""

    @staticmethod
    def get_model():
        if not GEMINI_API_KEY:
            return None
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            return genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini SDK: {e}")
            return None

    @classmethod
    def identify_vehicle(cls, input_data: Any, user_hints: str = "") -> Dict[str, Any]:
        if not ENABLE_AI_FEATURES:
            return {"error": "AI identification features are disabled via configuration."}

        model = cls.get_model()
        if not model:
            return {"error": "Gemini API key is not configured. Set GEMINI_API_KEY."}

        prompt = f"""
You are an expert automotive analyst. Identify the vehicle provided in the image or text query.
User Hints or context: "{user_hints}"

Return your output in STRICT valid JSON format with NO markdown wrapping, following this schema:
{{
  "brand": "Make or Brand Name",
  "model": "Model Name",
  "year": 2023,
  "body_type": "Sedan, SUV, Hatchback, etc.",
  "estimated_price": 5500000.0,
  "fuel_type": "Petrol / Diesel / Hybrid / Electric",
  "horsepower": "e.g. 180 hp",
  "key_features": "Feature 1, Feature 2, Feature 3",
  "notes": "2-3 sentences summarizing performance, trim notes, and market reputation."
}}
If the input is not a vehicle or cannot be identified, return:
{{
  "error": "Reason why the vehicle could not be identified."
}}
"""
        try:
            if isinstance(input_data, Image.Image):
                response = model.generate_content([prompt, input_data])
            elif isinstance(input_data, str):
                full_query = f"Vehicle Query: {input_data}\n\n{prompt}"
                response = model.generate_content(full_query)
            else:
                return {"error": "Unsupported input payload."}

            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]

            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {
                "brand": "Identified Car",
                "model": "Specification Result",
                "year": datetime.date.today().year,
                "body_type": "Sedan",
                "estimated_price": 0.0,
                "fuel_type": "Petrol",
                "horsepower": "N/A",
                "key_features": "Could not parse JSON structure.",
                "notes": response.text if "response" in locals() else "",
            }
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return {"error": f"AI service error: {str(e)}"}


# ==============================================================================
# 4. CONTROLLER & STATE INITIALIZATION
# ==============================================================================


def init_session_state() -> None:
    if "editing_car" not in st.session_state:
        st.session_state.editing_car = None
    if "pending_delete_id" not in st.session_state:
        st.session_state.pending_delete_id = None
    if "ai_detected_spec" not in st.session_state:
        st.session_state.ai_detected_spec = None


def reset_edit_state() -> None:
    st.session_state.editing_car = None


# ==============================================================================
# 5. USER INTERFACE COMPONENTS
# ==============================================================================


def main() -> None:
    st.set_page_config(page_title="Car Identifier & Inventory Tracker", page_icon="🚘", layout="wide")

    CarInventoryDatabase.initialize_db()
    init_session_state()

    st.title("🚘 Car Identifier & Fleet Tracker")
    st.caption("AI-powered car specification identifier and local inventory tracker powered by SQLite & Gemini Flash.")

    # Fetch stored vehicles
    try:
        cars = CarInventoryDatabase.get_all()
    except Exception as e:
        st.error(f"Database error: {e}")
        cars = []

    # KPI Summary Cards
    total_cars = len(cars)
    total_val = sum(float(c.get("price", 0.0)) for c in cars)
    avg_price = (total_val / total_cars) if total_cars > 0 else 0.0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Tracked Cars", total_cars)
    kpi2.metric("Portfolio Value", f"{DEFAULT_CURRENCY} {total_val:,.2f}")
    kpi3.metric("Average Valuation", f"{DEFAULT_CURRENCY} {avg_price:,.2f}")

    st.divider()

    # --------------------------------------------------------------------------
    # AI Identification Section
    # --------------------------------------------------------------------------
    st.subheader("🔍 AI Vehicle Identifier (Powered by Gemini Flash)")
    st.write("Upload an image, paste an image link, or type a car name to automatically fill out details.")

    tab_upload, tab_url, tab_text = st.tabs(["📷 Upload Car Image", "🔗 Image URL", "✍️ Car Name / Description"])

    target_image: Optional[Image.Image] = None
    text_desc: Optional[str] = None

    with tab_upload:
        up_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "webp"])
        if up_file:
            try:
                target_image = Image.open(up_file)
                st.image(target_image, caption="Vehicle Preview", width=360)
            except Exception as e:
                st.error(f"Image load failure: {e}")

    with tab_url:
        url_input = st.text_input("Paste Car Image URL", placeholder="[https://example.com/car.jpg](https://example.com/car.jpg)")
        if url_input.strip():
            try:
                res = requests.get(url_input.strip(), timeout=8)
                res.raise_for_status()
                target_image = Image.open(io.BytesIO(res.content))
                st.image(target_image, caption="Fetched Image", width=360)
            except Exception as e:
                st.error(f"Failed to fetch image: {e}")

    with tab_text:
        text_desc = st.text_area(
            "Enter Car Make / Model / Year description",
            placeholder="e.g. Honda Civic RS Turbo 2022 or Toyota Land Cruiser LC300 ZX",
        )

    ai_notes = st.text_input("Additional hints or trim questions (Optional)", placeholder="e.g. Pakistani domestic market edition")

    if st.button("🚀 Analyze & Fill Specifications", type="secondary"):
        payload = None
        if target_image:
            payload = target_image
        elif text_desc and text_desc.strip():
            payload = text_desc.strip()
        else:
            st.warning("Please provide an image or text description to analyze.")

        if payload:
            with st.spinner("Analyzing vehicle contours, badge, and engineering specifications..."):
                detected = GeminiCarAnalyst.identify_vehicle(payload, user_hints=ai_notes)
                if "error" in detected:
                    st.error(detected["error"])
                else:
                    st.session_state.ai_detected_spec = detected
                    st.success("Vehicle recognized! Values prefilled into the entry form below.")

    st.divider()

    # --------------------------------------------------------------------------
    # Car Entry & Modification Form
    # --------------------------------------------------------------------------
    is_editing = st.session_state.editing_car is not None
    form_title = "✏️ Edit Vehicle Record" if is_editing else "➕ Add Vehicle to Inventory"

    # Prefill precedence: Active editing > AI detected specs > Default empty
    active_record = st.session_state.editing_car or st.session_state.ai_detected_spec or {}

    with st.expander(form_title, expanded=is_editing or bool(st.session_state.ai_detected_spec)):
        with st.form("car_record_form", clear_on_submit=not is_editing):
            c_col1, c_col2, c_col3 = st.columns(3)

            with c_col1:
                brand_val = st.text_input("Brand / Make *", value=str(active_record.get("brand", "")))
                model_val = st.text_input("Model *", value=str(active_record.get("model", "")))
                year_default = int(active_record.get("year", datetime.date.today().year))
                year_val = st.number_input("Model Year *", min_value=1900, max_value=2030, value=year_default, step=1)

            with c_col2:
                body_type_str = str(active_record.get("body_type", SUPPORTED_BODY_TYPES[0]))
                body_idx = SUPPORTED_BODY_TYPES.index(body_type_str) if body_type_str in SUPPORTED_BODY_TYPES else 0
                body_val = st.selectbox("Body Type *", SUPPORTED_BODY_TYPES, index=body_idx)

                price_default = float(active_record.get("price") or active_record.get("estimated_price") or 0.0)
                price_val = st.text_input(f"Valuation / Price ({DEFAULT_CURRENCY}) *", value=str(price_default))

                fuel_str = str(active_record.get("fuel_type", FUEL_TYPES[0]))
                fuel_idx = FUEL_TYPES.index(fuel_str) if fuel_str in FUEL_TYPES else 0
                fuel_val = st.selectbox("Fuel Type *", FUEL_TYPES, index=fuel_idx)

            with c_col3:
                hp_val = st.text_input("Horsepower / Engine", value=str(active_record.get("horsepower", "")))
                features_val = st.text_input("Key Features", value=str(active_record.get("key_features", "")))
                notes_val = st.text_area("Notes / History", value=str(active_record.get("notes", "")))

            btn_label = "Update Car Record" if is_editing else "Save to Inventory"
            submitted = st.form_submit_button(btn_label, type="primary")

            if submitted:
                # Validation: Brand & Model
                if not brand_val.strip() or not model_val.strip():
                    st.error("Brand and Model fields are strictly required.")
                    st.stop()

                # Validation: Price Numeric
                try:
                    price_dec = Decimal(price_val.strip())
                    if price_dec < Decimal("0"):
                        st.error("Price cannot be negative.")
                        st.stop()
                except (InvalidOperation, ValueError):
                    st.error("Please enter a valid numeric valuation.")
                    st.stop()

                car_payload = {
                    "brand": brand_val.strip(),
                    "model": model_val.strip(),
                    "year": int(year_val),
                    "body_type": body_val,
                    "price": float(price_dec),
                    "currency": DEFAULT_CURRENCY,
                    "fuel_type": fuel_val,
                    "horsepower": hp_val.strip(),
                    "key_features": features_val.strip(),
                    "notes": notes_val.strip(),
                }

                try:
                    if is_editing:
                        CarInventoryDatabase.update(active_record["id"], car_payload)
                        st.success("Vehicle record updated.")
                        reset_edit_state()
                    else:
                        CarInventoryDatabase.create(car_payload)
                        st.success("Vehicle saved to inventory.")
                        st.session_state.ai_detected_spec = None
                    st.rerun()
                except Exception as ex:
                    st.error(f"Persistence error: {ex}")

        if is_editing:
            if st.button("Cancel Edit"):
                reset_edit_state()
                st.rerun()

    # --------------------------------------------------------------------------
    # Deletion Confirmation Guard
    # --------------------------------------------------------------------------
    if st.session_state.pending_delete_id:
        st.warning("⚠️ Are you sure you want to permanently delete this vehicle from inventory?")
        del_1, del_2, _ = st.columns([1, 1, 4])
        with del_1:
            if st.button("Yes, Delete", type="primary"):
                try:
                    CarInventoryDatabase.delete(st.session_state.pending_delete_id)
                    st.session_state.pending_delete_id = None
                    st.success("Vehicle deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Deletion failed: {e}")
        with del_2:
            if st.button("Cancel"):
                st.session_state.pending_delete_id = None
                st.rerun()

    # --------------------------------------------------------------------------
    # Inventory Records View, Filtering & Search
    # --------------------------------------------------------------------------
    st.subheader("📋 Tracked Vehicle Inventory")

    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        search_query = st.text_input("🔍 Search Brand, Model, or Notes", placeholder="e.g. Corolla, Civic, Turbo...")
    with s2:
        filter_body = st.selectbox("Filter Body Type", ["All"] + SUPPORTED_BODY_TYPES)
    with s3:
        sort_by = st.selectbox("Sort Order", ["Newest Added", "Year (New to Old)", "Price (High to Low)", "Price (Low to High)"])

    filtered_cars = list(cars)
    if filter_body != "All":
        filtered_cars = [c for c in filtered_cars if c.get("body_type") == filter_body]

    if search_query.strip():
        sq = search_query.strip().lower()
        filtered_cars = [
            c
            for c in filtered_cars
            if sq in str(c.get("brand", "")).lower()
            or sq in str(c.get("model", "")).lower()
            or sq in str(c.get("notes", "")).lower()
            or sq in str(c.get("key_features", "")).lower()
        ]

    if sort_by == "Newest Added":
        filtered_cars.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    elif sort_by == "Year (New to Old)":
        filtered_cars.sort(key=lambda x: int(x.get("year", 0)), reverse=True)
    elif sort_by == "Price (High to Low)":
        filtered_cars.sort(key=lambda x: float(x.get("price", 0.0)), reverse=True)
    elif sort_by == "Price (Low to High)":
        filtered_cars.sort(key=lambda x: float(x.get("price", 0.0)))

    if not filtered_cars:
        st.info("No vehicles match the selected criteria.")
    else:
        for car in filtered_cars:
            row_1, row_2, row_3, row_4, row_5, row_6 = st.columns([1, 2.5, 1.5, 2, 0.8, 0.8])
            row_1.write(f"📅 **{car.get('year')}**")
            row_2.write(f"**{car.get('brand')} {car.get('model')}**\n*{car.get('notes') or '-'}*")
            row_3.caption(f"🏷️ {car.get('body_type')} | ⛽ {car.get('fuel_type')}\n🐎 {car.get('horsepower') or 'N/A'}")
            row_4.write(f"**{car.get('currency')} {float(car.get('price', 0.0)):,.2f}**")

            with row_5:
                if st.button("Edit", key=f"edit_btn_{car['id']}"):
                    st.session_state.editing_car = car
                    st.rerun()

            with row_6:
                if st.button("Delete", key=f"del_btn_{car['id']}"):
                    st.session_state.pending_delete_id = car["id"]
                    st.rerun()

            st.divider()


if __name__ == "__main__":
    main()
