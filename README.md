# 🚘 Car Identifier & Fleet Inventory Tracker

An intelligent vehicle identification and inventory management web application built with **Streamlit**, an embedded **SQLite** database, and **Google Gemini 1.5 Flash (`google-generativeai`)**.

---

## 🌟 Key Capabilities

- **Multimodal AI Analysis:** Identify brand, model, generation, horsepower, and features from photo uploads, image URLs, or text queries.
- **Embedded Local Persistence:** Runs out of the box with SQLite. No Firebase credentials, service account JSON files, or permission configurations needed.
- **Full Inventory CRUD:** Record, browse, filter, edit, and safely delete vehicle entries.
- **Search & Multi-Facet Filters:** Filter by body style (Sedan, SUV, Hatchback, etc.), search by trim notes, and sort by year or valuation.
- **Configurable Base Currency:** Default base currency configured for `PKR`.

---

## 🛠️ Local Installation

### 1. Set Up Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
