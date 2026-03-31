# Geo Crop Advisory

Location-aware crop recommendation with a **FastAPI** prediction service and a **Streamlit** chat-style UI. The backend finds the nearest district from soil data, blends weather from [Open-Meteo](https://open-meteo.com/) (no API key), runs a trained Random Forest model, and optionally enriches output with multilingual summaries via **Groq**.

Repository: [rajiv-golait/geo-crop-advisory](https://github.com/rajiv-golait/geo-crop-advisory)

## Stack

- Python 3.10+
- **FastAPI** + **Uvicorn** — `/predict` from latitude/longitude
- **Streamlit** — UI that geocodes a place and calls the API
- **scikit-learn** / **joblib** — saved model and label encoder in `models/`
- **python-dotenv** — local configuration

## Quick start

1. Clone the repository and enter the project directory.

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy environment template (optional but recommended):

   ```bash
   copy .env.example .env
   ```

   Edit `.env` as needed. See **Configuration** below.

4. Start the API (from the project root):

   ```bash
   uvicorn backend_api:app --reload
   ```

5. In a second terminal, start the UI:

   ```bash
   streamlit run app.py
   ```

6. Open the Streamlit URL in your browser (usually `http://localhost:8501`). Enter a place name; the app geocodes it and requests a recommendation from your local API.

## Configuration

| Variable | Purpose |
| -------- | ------- |
| `PREDICT_API_BASE` | Base URL of the FastAPI app (default in example: `http://127.0.0.1:8000`). |
| `GROQ_API_KEY` | Optional. Enables richer LLM-backed farmer chat / localized cards. Without it, core prediction still works. |

Weather is fetched from Open-Meteo; no weather API key is required.

## Project layout

- `backend_api.py` — FastAPI app: nearest district + N/P/K mapping + Open-Meteo + model inference
- `app.py` — Streamlit UI
- `data/` — CSV datasets used for district/soil context
- `models/` — `random_forest_model.pkl`, `label_encoder.pkl`
- `utils/` — prediction helpers, weather, i18n, optional LLM integration
- `Crop_Recommendation.ipynb` — notebook used in model/dataset workflow

## License

If you add a license (e.g. MIT), place it in `LICENSE` and reference it here.
