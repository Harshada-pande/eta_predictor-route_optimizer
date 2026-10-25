# Delivery ETA Predictor and Route Optimizer

An AI/ML-based delivery optimization system that:

1. Predicts delivery time (ETA) using **XGBoost**.
2. Builds an ETA matrix between the depot and delivery stops.
3. Finds an optimized delivery sequence using **Held-Karp** or **Hill Climbing**.
4. Provides the functionality through a **FastAPI backend**.
5. Can be connected to a **Streamlit frontend**.

---

## 1. Project Architecture

```text
Streamlit Frontend
        |
        | HTTP POST
        v
FastAPI Backend
        |
        v
Feature Builder
        |
        v
XGBoost ETA Model
        |
        v
ETA Matrix
        |
        v
Route Optimizer
   /           \
Held-Karp   Hill Climbing
        |
        v
Optimized Route + ETA
```

---

# 2. Requirements

Make sure you have:

- Python **3.12**
- Git
- VS Code (recommended)

Python 3.13 may cause package compatibility issues. **Python 3.12 is recommended for this project.**

---

# 3. Clone the Project

Open PowerShell or the VS Code terminal.

Go to the folder where you want to keep the project:

```powershell
cd Downloads
```

Clone the repository:

```powershell
git clone https://github.com/Harshada-pande/eta_predictor-route_optimizer.git
```

Enter the project:

```powershell
cd eta_predictor-route_optimizer
```

Open it in VS Code:

```powershell
code .
```

If `code .` does not work, simply open VS Code and select:

**File → Open Folder → eta_predictor-route_optimizer**

---

# 4. Create a Virtual Environment

Inside the project folder, run:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see something similar to:

```text
(.venv) PS C:\...\eta_predictor-route_optimizer>
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 5. Install Backend Dependencies

Install the core ML/backend dependencies:

```powershell
pip install -r requirements.txt
```

For the FastAPI and Streamlit application:

```powershell
pip install -r requirements-app.txt
```

---

# 6. Verify the Installation

Run the complete test suite:

```powershell
python -m pytest -q
```

Expected result:

```text
101 passed
```

There may be one Starlette/httpx deprecation warning. It does not prevent the project from running.

---

# 7. Start the FastAPI Backend

From the project root, run:

```powershell
python -m uvicorn api.main:app --reload
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal **running**.

---

# 8. Check the API

Open your browser:

```text
http://127.0.0.1:8000/docs
```

This opens the FastAPI Swagger documentation.

You should see:

```text
GET  /health
POST /optimize
```

---

# 9. Check Backend Health

Open a **second terminal** in the project folder.

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```text
status
------
ok
```

---

# 10. API Usage

The main endpoint is:

```text
POST /optimize
```

It accepts:

### Depot

```text
depot_lat
depot_lon
```

### Rider-level information

```text
vehicle_type
vehicle_condition
delivery_person_age
delivery_person_rating
festival
city_type
```

### Stop-level information

```text
id
lat
lon
weather
traffic
multiple_deliveries
```

### Other inputs

```text
departure_time
algorithm
```

---

# 11. Example API Request

Example:

```json
{
  "depot_lat": 18.5204,
  "depot_lon": 73.8567,
  "rider": {
    "vehicle_type": "Motorcycle",
    "vehicle_condition": 2,
    "delivery_person_age": 25,
    "delivery_person_rating": 4.5
  },
  "festival": "No",
  "city_type": "Metropolitan",
  "stops": [
    {
      "id": "A",
      "lat": 18.5304,
      "lon": 73.8467,
      "weather": "Sunny",
      "traffic": "Medium",
      "multiple_deliveries": 0
    },
    {
      "id": "B",
      "lat": 18.5104,
      "lon": 73.8667,
      "weather": "Cloudy",
      "traffic": "High",
      "multiple_deliveries": 1
    }
  ],
  "departure_time": "2026-10-25T09:00:00",
  "algorithm": "held_karp"
}
```

---

# 12. Available Optimization Algorithms

## Held-Karp

```text
algorithm = "held_karp"
```

Held-Karp performs an exact search for the best route.

It is suitable for a relatively small number of delivery stops.

## Hill Climbing

```text
algorithm = "hill_climbing"
```

Hill Climbing is a heuristic optimization method.

It is generally faster but can get stuck in a local optimum.

---

# 13. Example Response

The API returns information such as:

```text
algorithm: held_karp

route:
Depot → B → A

total_minutes:
40.58

baseline_total_minutes:
40.67

improvement_pct:
0.22
```

It also returns ETA and arrival time for each stop.

---

# 14. Running the Streamlit Frontend

The Streamlit frontend should communicate with the FastAPI backend.

First make sure the FastAPI backend is running:

```powershell
python -m uvicorn api.main:app --reload
```

Keep that terminal open.

Open a **second terminal**.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then start Streamlit:

```powershell
streamlit run frontend/app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

---

# 15. Important: Backend Must Be Running

The frontend communicates with:

```text
http://127.0.0.1:8000
```

Therefore, if you open the Streamlit frontend without starting FastAPI first, the frontend will not be able to get ETA predictions or route optimization results.

Always run:

### Terminal 1

```powershell
python -m uvicorn api.main:app --reload
```

### Terminal 2

```powershell
streamlit run frontend/app.py
```

---

# 16. Important Project Structure

```text
eta_predictor-route_optimizer/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── configs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── evaluation/
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── artifacts/
│   │   ├── best_eta_model.joblib
│   │   └── feature_columns.json
│   └── train.py
│
├── optimizer/
│   ├── hill_climbing.py
│   └── held_karp.py
│
├── preprocessing/
│   ├── cleaning.py
│   ├── feature_builder.py
│   └── prepare_data.py
│
├── tests/
│
├── requirements.txt
├── requirements-app.txt
└── README.md
```

---

# 17. Model Performance

The final XGBoost model achieved the following held-out test performance:

| Metric | Result |
|---|---:|
| MAE | 3.09 min |
| RMSE | 3.87 min |
| R² | 0.8271 |
| R² as percentage | 82.71% |
| Delay Classification Accuracy | 93.17% |
| Delay Classification F1 | 87.14% |

The R² value of **0.8271** means the model explains approximately **82.71% of the variation in delivery time in the held-out dataset**.

---

# 18. Route Optimization Benchmark

The optimization algorithms were benchmarked on 100 synthetic cases with 10 delivery stops.

Results:

| Metric | Result |
|---|---:|
| Hill Climbing suboptimal cases | 94/100 |
| Mean optimality gap | 0.77% |
| Maximum gap | 5.13% |
| Hill Climbing mean runtime | 3.624 ms |
| Held-Karp mean runtime | 26.831 ms |

Held-Karp is slower but provides an exact solution for the tested problem size.

---

# 19. Common Problems

### Problem: `python` is not recognized

Install Python 3.12 and make sure Python is added to PATH.

Check:

```powershell
python --version
```

Expected:

```text
Python 3.12.x
```

---

### Problem: Virtual environment does not activate

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### Problem: API connection error in Streamlit

Make sure FastAPI is running:

```powershell
python -m uvicorn api.main:app --reload
```

Then check:

```text
http://127.0.0.1:8000/docs
```

---

### Problem: Port 8000 is already in use

Stop the previous FastAPI process or use another terminal where the existing server is already running.

---

### Problem: Tests fail

Run:

```powershell
python -m pytest -q
```

Read the first failing test before making any changes.

Do not modify the model or preprocessing files just to make a test pass.

---

# 20. Recommended Startup Sequence

Every time you open the project:

### Terminal 1

```powershell
cd path\to\eta_predictor-route_optimizer
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

### Terminal 2

```powershell
cd path\to\eta_predictor-route_optimizer
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Then open the Streamlit URL shown in the terminal.

---

# 21. For Developers Working on the Frontend

The frontend should communicate with the backend through:

```text
POST /optimize
```

Do **not** directly import:

```text
models
optimizer
preprocessing
```

into the Streamlit frontend.

The intended architecture is:

```text
Streamlit
    ↓
HTTP POST /optimize
    ↓
FastAPI
    ↓
Feature Builder
    ↓
XGBoost
    ↓
ETA Matrix
    ↓
Route Optimizer
    ↓
JSON Response
    ↓
Streamlit
```

This keeps the frontend and backend separated.

---

# 22. Git Workflow

Before making changes:

```powershell
git pull
```

Check the current status:

```powershell
git status
```

After making frontend changes:

```powershell
git add frontend/
git commit -m "Update Streamlit frontend"
git push
```

---

## Project Status

### Backend

- [x] Data preprocessing
- [x] Feature engineering
- [x] XGBoost ETA prediction
- [x] ETA matrix
- [x] Hill Climbing
- [x] Held-Karp
- [x] FastAPI API
- [x] API testing
- [x] Automated tests

### Frontend

- [ ] Streamlit UI
- [ ] FastAPI integration
- [ ] Final end-to-end testing

---

## Quick Start

If everything is already installed, the fastest way to run the project is:

**Terminal 1:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

**Terminal 2:**

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Then open the Streamlit URL shown in the terminal.
