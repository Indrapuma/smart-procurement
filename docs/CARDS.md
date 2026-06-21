# Implementation Cards — Agentic AI Purchase Decision Assistant

> Urutan card mengikuti dependency: setiap card bisa dikerjakan setelah card sebelumnya selesai.  
> Total: 19 card, 8 fase.

---

## FASE 0 — Project Setup

---

### CARD-00 · Project Initialization

**Tujuan:** Siapkan struktur folder, environment, dan dependency dasar sebelum kode apapun ditulis.

**Steps:**
1. Buat folder root project dan struktur direktori:
   ```
   procurement-agent/
   ├── agent/           # LangGraph nodes & graph
   ├── tools/           # SQL tools
   ├── ml/              # model training & SHAP
   ├── api/             # FastAPI
   ├── frontend/        # Streamlit
   ├── data/            # raw & processed dataset
   ├── db/              # migration scripts
   ├── tests/
   └── docs/
   ```
2. Inisialisasi Python environment (`uv` atau `venv`)
3. Buat `pyproject.toml` / `requirements.txt` dengan dependency awal:
   - `langgraph`, `langchain-google-genai`
   - `fastapi`, `uvicorn`
   - `sqlalchemy`, `psycopg2-binary`
   - `xgboost`, `shap`, `scikit-learn`, `pandas`, `numpy`
   - `streamlit`
   - `python-dotenv`
4. Buat `.env.example`:
   ```
   DATABASE_URL=postgresql://user:pass@localhost:5432/procurement
   GOOGLE_API_KEY=          # dari Google AI Studio: aistudio.google.com
   LANGSMITH_API_KEY=       # opsional, untuk tracing
   LANGSMITH_PROJECT=procurement-agent
   ```
5. Buat `.gitignore` (exclude `.env`, `data/raw/`, `__pycache__`, model binaries)
6. `git init` + initial commit

**Output:** Repo siap, environment aktif, semua dependency ter-install.

---

## FASE 1 — Database & Data Pipeline

---

### CARD-01 · Database Schema Setup

**Tujuan:** Buat database PostgreSQL dengan 4 tabel sesuai skema PRD.

**Dependencies:** CARD-00

**Steps:**
1. Buat database `procurement` di PostgreSQL lokal (atau Supabase/Neon untuk langsung cloud)
2. Tulis migration script `db/001_create_tables.sql`:
   ```sql
   CREATE TABLE products (
       product_id   VARCHAR(50) PRIMARY KEY,
       name         VARCHAR(200) NOT NULL,
       category     VARCHAR(100),
       unit_price   NUMERIC(10,2)
   );

   CREATE TABLE sales (
       sale_id     SERIAL PRIMARY KEY,
       product_id  VARCHAR(50) REFERENCES products(product_id),
       date        DATE NOT NULL,
       qty_sold    INTEGER NOT NULL
   );

   CREATE TABLE inventory (
       product_id    VARCHAR(50) PRIMARY KEY REFERENCES products(product_id),
       current_stock INTEGER NOT NULL,
       last_updated  TIMESTAMP DEFAULT NOW()
   );

   CREATE TABLE suppliers (
       supplier_id    SERIAL PRIMARY KEY,
       product_id     VARCHAR(50) REFERENCES products(product_id),
       lead_time_days INTEGER NOT NULL,
       min_order_qty  INTEGER DEFAULT 1
   );
   ```
3. Tambahkan index untuk query yang sering dipakai:
   ```sql
   CREATE INDEX idx_sales_product_date ON sales(product_id, date);
   ```
4. Jalankan migration dan verifikasi semua tabel terbuat

**Output:** Database siap dengan 4 tabel + index.

---

### CARD-02 · ETL Pipeline (Kaggle → PostgreSQL)

**Tujuan:** Download dataset Kaggle, transform ke skema internal, load ke database.

**Dependencies:** CARD-01

**Dataset:** Retail Store Inventory Forecasting Dataset (Kaggle — ~73K baris)  
Kolom tersedia: `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Demand Forecast`, `Weather Condition`, `Holiday/Promotion`, `Price`

**Steps:**
1. Download dataset via Kaggle API:
   ```bash
   kaggle datasets download anirudhchauhan/retail-store-inventory-forecasting-dataset
   ```
2. Buat script `data/etl.py` dengan fungsi:
   - `extract()` — load CSV ke DataFrame
   - `transform_products()` — deduplicate Product ID + Category + hitung avg price
   - `transform_sales()` — map ke tabel `sales` (product_id, date, qty_sold)
   - `transform_inventory()` — ambil stok paling akhir per produk (last date)
   - `transform_suppliers()` — generate lead time sintetis:
     - Elektronik: random 14–21 hari
     - Makanan: random 3–7 hari
     - Lainnya: random 7–14 hari
     - `min_order_qty`: random 10–50 unit per kategori
3. `load()` — insert ke PostgreSQL via SQLAlchemy
4. Tambahkan kolom tambahan dari dataset ke tabel `sales` untuk XAI (simpan sebagai tabel `sales_context`):
   ```sql
   CREATE TABLE sales_context (
       sale_id              INTEGER REFERENCES sales(sale_id),
       weather_condition    VARCHAR(50),
       is_holiday_promo     BOOLEAN,
       price_at_sale        NUMERIC(10,2)
   );
   ```
5. Verifikasi row count dan sample data

**Output:** ~73K baris sales, products, inventory current, suppliers sintetis — semua ada di PostgreSQL.

---

## FASE 2 — SQL Tools

---

### CARD-03 · SQL Tools Implementation

**Tujuan:** Buat 6 SQL tools yang akan dipanggil agent. Semua read-only, parameterized, whitelist tabel.

**Dependencies:** CARD-01, CARD-02

**File:** `tools/sql_tools.py`

**Steps:**
1. Buat `DatabaseConnection` class dengan SQLAlchemy engine (singleton pattern)
2. Implementasi 6 tool functions, masing-masing return dict:

   **`get_current_stock(product_id: str) → dict`**
   ```sql
   SELECT i.current_stock, i.last_updated, p.name, p.category
   FROM inventory i JOIN products p USING(product_id)
   WHERE i.product_id = :product_id
   ```

   **`get_sales_history(product_id: str, n_months: int) → dict`**
   ```sql
   SELECT date, qty_sold FROM sales
   WHERE product_id = :product_id
     AND date >= CURRENT_DATE - INTERVAL ':n_months months'
   ORDER BY date
   ```

   **`get_supplier_leadtime(product_id: str) → dict`**
   ```sql
   SELECT lead_time_days, min_order_qty FROM suppliers
   WHERE product_id = :product_id LIMIT 1
   ```

   **`get_demand_trend(product_id: str, n_months: int) → dict`**  
   Hitung di Python (bukan SQL): growth rate MoM, moving average 4-week dari `get_sales_history`

   **`get_seasonality_index(product_id: str) → dict`**  
   Hitung di Python: rata-rata qty per bulan dibagi rata-rata keseluruhan (index musiman per bulan)

   **`rank_products_by_risk(category: str = None) → list`**  
   ```sql
   SELECT p.product_id, p.name, p.category,
          i.current_stock,
          s.lead_time_days
   FROM products p
   JOIN inventory i USING(product_id)
   JOIN suppliers s USING(product_id)
   WHERE (:category IS NULL OR p.category = :category)
   ```
   Lalu hitung risk score di Python: `days_of_stock = current_stock / avg_daily_sales`

3. Buat whitelist validator — hanya boleh query tabel: `sales`, `inventory`, `products`, `suppliers`, `sales_context`
4. Bungkus setiap function sebagai LangChain `@tool` dengan docstring yang jelas (agent pakai docstring untuk routing)
5. Tulis unit test `tests/test_sql_tools.py` untuk setiap tool

**Output:** 6 tools siap dipanggil, semua tested.

---

## FASE 3 — ML Model & XAI

---

### CARD-04 · Feature Engineering

**Tujuan:** Transform data mentah dari database menjadi feature matrix siap training XGBoost.

**Dependencies:** CARD-02

**File:** `ml/features.py`

**Steps:**
1. Load data dari PostgreSQL (join `sales` + `sales_context` + `products`)
2. Buat fungsi `build_feature_matrix(product_id=None)` yang menghasilkan DataFrame dengan kolom:

   | Feature | Cara dapat |
   |---|---|
   | `month` | dari kolom `date` |
   | `week_of_year` | dari kolom `date` |
   | `is_holiday_promo` | dari `sales_context` |
   | `weather_encoded` | label encode `weather_condition` |
   | `price` | dari `sales_context.price_at_sale` |
   | `lag_1_week` | qty_sold 7 hari lalu |
   | `lag_4_week` | qty_sold 28 hari lalu |
   | `rolling_avg_4w` | moving average 4 minggu |
   | `rolling_std_4w` | standar deviasi 4 minggu |
   | `category_encoded` | label encode `category` |
   | `target` | `qty_sold` (yang mau diprediksi) |

3. Handle missing values (lag features akan NaN di awal)
4. Simpan `LabelEncoder` mapping ke file JSON (dibutuhkan saat inference)

**Output:** `ml/feature_matrix.parquet` + `ml/encoders.json`

---

### CARD-05 · XGBoost Model Training

**Tujuan:** Train model forecasting demand per kategori, evaluasi dengan MAPE, simpan model.

**Dependencies:** CARD-04

**File:** `ml/train.py`

**Steps:**
1. Load feature matrix dari CARD-04
2. Split: 80% train (chronological), 20% test — **jangan random shuffle** karena ini time series
3. Train `XGBRegressor` per kategori produk:
   ```python
   params = {
       "n_estimators": 300,
       "max_depth": 5,
       "learning_rate": 0.05,
       "subsample": 0.8,
       "colsample_bytree": 0.8,
       "objective": "reg:squarederror"
   }
   ```
4. Evaluasi dengan MAPE:
   ```python
   mape = mean_absolute_percentage_error(y_test, y_pred)
   ```
   Target: MAPE < 20%
5. Simpan model per kategori: `ml/models/{category}_model.json`
6. Simpan metadata: `ml/models/metadata.json` (MAPE per kategori, tanggal training, feature list)

**Output:** Model files tersimpan, MAPE tercapai < 20%.

---

### CARD-06 · SHAP Integration & Explanation Templates

**Tujuan:** Hitung SHAP values per prediksi dan terjemahkan ke teks penjelasan yang bisa dipahami non-teknis.

**Dependencies:** CARD-05

**File:** `ml/explainer.py`

**Steps:**
1. Load model XGBoost dan buat `shap.TreeExplainer`
2. Buat fungsi `explain_prediction(product_id, features_row) → dict`:
   - Jalankan SHAP: `shap_values = explainer.shap_values(features_row)`
   - Return dict: `{feature_name: shap_value}` diurutkan dari absolute terbesar
3. Buat rule-based template translator `shap_to_text(shap_dict) → list[str]`:

   | Feature | Teks jika positif | Teks jika negatif |
   |---|---|---|
   | `rolling_avg_4w` | "Tren penjualan 4 minggu terakhir meningkat" | "Tren penjualan 4 minggu terakhir menurun" |
   | `is_holiday_promo` | "Ada promosi/hari libur yang meningkatkan permintaan" | "Tidak ada promosi aktif" |
   | `weather_encoded` | "Kondisi cuaca mendukung penjualan" | "Kondisi cuaca menghambat penjualan" |
   | `month` | "Bulan ini termasuk musim permintaan tinggi" | "Bulan ini termasuk musim permintaan rendah" |
   | `lag_1_week` | "Minggu lalu penjualan lebih tinggi dari biasanya" | "Minggu lalu penjualan lebih rendah dari biasanya" |
   | `price` | — | "Kenaikan harga menekan permintaan" |

4. Output akhir: list bullet points (max 3 faktor teratas), siap dikirim ke `polish_node`

**Output:** Fungsi explain yang menghasilkan daftar faktor terstruktur per prediksi.

---

## FASE 4 — LangGraph Agent

---

### CARD-07 · State Schema & Graph Setup

**Tujuan:** Definisikan state agent dan skeleton graph LangGraph sebelum isi node.

**Dependencies:** CARD-00

**File:** `agent/state.py`, `agent/graph.py`

**Steps:**
1. Definisikan `AgentState` di `agent/state.py`:
   ```python
   from typing import TypedDict, Optional

   class AgentState(TypedDict):
       user_query: str
       intent: str                  # simple_lookup | purchase_recommendation | explain_trend | risk_ranking
       product_id: Optional[str]
       category: Optional[str]
       sales_data: Optional[dict]
       stock_data: Optional[dict]
       forecast: Optional[dict]
       shap_values: Optional[dict]
       explanation_draft: Optional[str]
       recommendation: Optional[dict]
       final_answer: Optional[str]
       error: Optional[str]
   ```
2. Buat `agent/graph.py` — susun skeleton graph:
   ```python
   from langgraph.graph import StateGraph, END

   graph = StateGraph(AgentState)
   graph.add_node("router", router_node)
   graph.add_node("fetch_data", fetch_data_node)
   graph.add_node("forecast", forecast_node)
   graph.add_node("explain", explain_node)
   graph.add_node("decision", decision_node)
   graph.add_node("polish", polish_node)

   graph.set_entry_point("router")
   # Edges dikonfigurasi di card masing-masing node
   ```
3. Verifikasi graph dapat di-compile tanpa error

**Output:** State schema + graph skeleton ter-compile.

---

### CARD-08 · Router Node

**Tujuan:** Klasifikasi intent dari query user menggunakan LLM, ekstrak product_id jika ada.

**Dependencies:** CARD-07

**File:** `agent/nodes/router.py`

**Steps:**
1. Buat prompt klasifikasi:
   ```
   Klasifikasikan pertanyaan berikut ke salah satu intent:
   - simple_lookup: cek stok saat ini
   - purchase_recommendation: rekomendasi berapa unit yang perlu dibeli
   - explain_trend: penjelasan kenapa penjualan naik/turun
   - risk_ranking: produk mana yang paling berisiko stockout/overstock

   Ekstrak juga product_id jika disebutkan (format: P-XXX).
   Jika tidak ada product_id spesifik, return null.

   Query: {user_query}

   Jawab dalam JSON: {"intent": "...", "product_id": "..." atau null, "category": "..." atau null}
   ```
2. Gunakan `gemini-2.0-flash` (cepat dan gratis untuk klasifikasi)
3. Parse JSON response, update state: `intent`, `product_id`, `category`
4. Tambahkan conditional edges di graph berdasarkan intent:
   - `simple_lookup` → `fetch_data` → `polish` → END
   - `purchase_recommendation` → `fetch_data` → `forecast` → `explain` → `decision` → `polish` → END
   - `explain_trend` → `fetch_data` → `forecast` → `explain` → `polish` → END
   - `risk_ranking` → `fetch_data` → `decision` → `polish` → END
5. Fallback: jika JSON tidak valid, tanya klarifikasi ke user

**Output:** Router memilih jalur yang tepat berdasarkan intent.

---

### CARD-09 · Fetch Data Node

**Tujuan:** Panggil SQL tools yang tepat sesuai intent, simpan hasilnya ke state.

**Dependencies:** CARD-03, CARD-07

**File:** `agent/nodes/fetch_data.py`

**Steps:**
1. Baca `state["intent"]` dan `state["product_id"]`
2. Jalankan tools berdasarkan intent:
   - `simple_lookup`:
     - `get_current_stock(product_id)` → `state["stock_data"]`
   - `purchase_recommendation`:
     - `get_current_stock(product_id)` → `state["stock_data"]`
     - `get_sales_history(product_id, n_months=6)` → `state["sales_data"]`
     - `get_supplier_leadtime(product_id)` → masuk ke `state["stock_data"]["leadtime"]`
   - `explain_trend`:
     - `get_sales_history(product_id, n_months=3)` → `state["sales_data"]`
     - `get_demand_trend(product_id, n_months=3)` → masuk ke `state["sales_data"]["trend"]`
   - `risk_ranking`:
     - `rank_products_by_risk(category=state["category"])` → `state["sales_data"]`
3. Handle error: jika product_id tidak ditemukan, set `state["error"]` dan skip ke `polish`

**Output:** State berisi data lengkap yang dibutuhkan node berikutnya.

---

### CARD-10 · Forecast Node

**Tujuan:** Jalankan XGBoost model untuk prediksi demand periode berikutnya.

**Dependencies:** CARD-05, CARD-07

**File:** `agent/nodes/forecast.py`

**Steps:**
1. Ambil `sales_data` dari state, bangun feature row untuk prediksi
2. Load model sesuai kategori produk
3. Prediksi demand untuk periode `lead_time` ke depan:
   - Prediksi per hari × `lead_time_days` = total forecasted demand selama lead time
4. Hitung confidence interval sederhana (± 1 std dari residual training)
5. Update state:
   ```python
   state["forecast"] = {
       "predicted_demand": 120,
       "confidence_low": 105,
       "confidence_high": 135,
       "period_days": 14,   # = lead_time
       "model_mape": 0.14
   }
   ```

**Output:** Prediksi demand tersimpan di state dengan confidence interval.

---

### CARD-11 · Explain Node

**Tujuan:** Hitung SHAP values dan terjemahkan ke draft penjelasan berbahasa Indonesia.

**Dependencies:** CARD-06, CARD-07

**File:** `agent/nodes/explain.py`

**Steps:**
1. Ambil feature row yang sama dengan CARD-10
2. Jalankan `explain_prediction()` dari `ml/explainer.py`
3. Jalankan `shap_to_text()` untuk konversi ke bullet points
4. Susun `explanation_draft` sebagai string terstruktur:
   ```
   Faktor utama yang mempengaruhi prediksi:
   - Tren penjualan 4 minggu terakhir meningkat (+12%)
   - Ada promosi aktif bulan ini
   - Bulan ini termasuk musim permintaan tinggi

   Prediksi demand: 120 unit (rentang: 105–135 unit)
   ```
5. Update `state["explanation_draft"]` dan `state["shap_values"]`

**Output:** Draft penjelasan siap dikirim ke polish node.

---

### CARD-12 · Decision Node

**Tujuan:** Hitung rekomendasi qty pembelian (formula EOQ/reorder point) atau risk ranking.

**Dependencies:** CARD-07

**File:** `agent/nodes/decision.py`

**Steps:**
1. Untuk intent `purchase_recommendation`:
   - Ambil dari state: `forecast["predicted_demand"]`, `stock_data["current_stock"]`, `stock_data["leadtime"]["lead_time_days"]`
   - Hitung safety stock:
     ```python
     z_score = 1.65  # service level 95%
     std_demand = std(sales_history_daily)
     safety_stock = z_score * std_demand * sqrt(lead_time_days)
     ```
   - Hitung reorder qty:
     ```python
     reorder_qty = max(0, forecasted_demand + safety_stock - current_stock)
     reorder_qty = max(reorder_qty, min_order_qty)  # floor ke MOQ
     ```
   - Update state:
     ```python
     state["recommendation"] = {
         "reorder_qty": 120,
         "safety_stock": 25,
         "min_order_qty": 10,
         "current_stock": 35,
         "forecasted_demand": 110
     }
     ```
2. Untuk intent `risk_ranking`:
   - Hitung `days_of_stock = current_stock / avg_daily_demand` untuk setiap produk
   - Klasifikasikan:
     - `days_of_stock < lead_time_days` → **CRITICAL** (akan stockout sebelum barang datang)
     - `days_of_stock < lead_time_days * 1.5` → **WARNING**
     - `current_stock > avg_monthly_demand * 3` → **OVERSTOCK**
     - Lainnya → **OK**
   - Urutkan: CRITICAL → WARNING → OVERSTOCK → OK
   - Update `state["recommendation"]` dengan list berurutan

**Output:** Angka rekomendasi atau ranking tersimpan di state.

---

### CARD-13 · Polish Node

**Tujuan:** LLM merangkai semua informasi dari state menjadi jawaban final yang natural dan mudah dipahami non-teknis.

**Dependencies:** CARD-07

**File:** `agent/nodes/polish.py`

**Steps:**
1. Susun prompt berdasarkan intent:
   - Untuk `simple_lookup`: langsung format stok dari `stock_data`
   - Untuk `purchase_recommendation`:
     ```
     Kamu adalah asisten procurement yang menjelaskan rekomendasi pembelian kepada manajer non-teknis.

     Data:
     - Produk: {name}
     - Stok saat ini: {current_stock} unit
     - Lead time supplier: {lead_time_days} hari
     - Prediksi demand selama lead time: {forecasted_demand} unit
     - Safety stock: {safety_stock} unit
     - Rekomendasi beli: {reorder_qty} unit

     Alasan (dari analisis data):
     {explanation_draft}

     Tulis jawaban dalam 3–5 kalimat, gunakan bahasa Indonesia yang jelas.
     Sertakan angka-angka di atas PERSIS tanpa mengubahnya.
     Jangan gunakan istilah teknis ML.
     ```
2. Gunakan `gemini-2.5-flash` untuk polish (kualitas narasi)
3. Validasi pasca-polish: angka numerik dalam `recommendation` harus muncul di `final_answer` — jika tidak, log warning
4. Update `state["final_answer"]`

**Output:** Jawaban final siap ditampilkan ke user.

---

### CARD-14 · Risk Ranking Node (Full Flow Test)

**Tujuan:** Verifikasi seluruh graph berjalan end-to-end untuk semua 4 intent.

**Dependencies:** CARD-08 s/d CARD-13

**File:** `tests/test_graph_e2e.py`

**Steps:**
1. Buat test set 10 pertanyaan (minimal 2 per intent):
   ```python
   test_cases = [
       {"query": "Stok produk P-001 tinggal berapa?", "expected_intent": "simple_lookup"},
       {"query": "Berapa unit P-003 yang perlu dibeli bulan depan?", "expected_intent": "purchase_recommendation"},
       {"query": "Kenapa penjualan P-007 turun bulan ini?", "expected_intent": "explain_trend"},
       {"query": "Produk mana yang paling berisiko habis stok?", "expected_intent": "risk_ranking"},
       # ... dst
   ]
   ```
2. Jalankan graph untuk setiap test case, ukur:
   - Routing accuracy (intent benar atau tidak)
   - Response time
   - `final_answer` tidak kosong
3. Log semua trace ke LangSmith

**Acceptance criteria:**
- Routing accuracy ≥ 90%
- `simple_lookup` < 2 detik
- `purchase_recommendation` < 8 detik

**Output:** Graph verified end-to-end, semua intent berjalan benar.

---

## FASE 5 — Backend API

---

### CARD-15 · FastAPI Backend

**Tujuan:** Buat REST API yang menerima query user dan mengembalikan hasil dari LangGraph agent.

**Dependencies:** CARD-14

**File:** `api/main.py`, `api/routers/chat.py`

**Steps:**
1. Setup FastAPI app dengan CORS (untuk frontend Streamlit/React)
2. Buat endpoint utama:
   ```
   POST /chat
   Body: {"query": "Berapa unit P-001 yang perlu dibeli?", "session_id": "..."}
   Response: {
       "answer": "...",
       "intent": "purchase_recommendation",
       "trace_url": "https://smith.langchain.com/...",  # opsional
       "response_time_ms": 3421
   }
   ```
3. Buat endpoint health check:
   ```
   GET /health → {"status": "ok", "db": "connected", "model": "loaded"}
   ```
4. Rate limiting: max 10 request/menit per IP (pakai `slowapi`)
5. Error handling: return 400 jika query kosong, 500 jika agent error dengan pesan yang user-friendly
6. Tambahkan logging request/response (tanpa log konten sensitif)

**Output:** API berjalan di `localhost:8000`, siap dihit dari frontend.

---

## FASE 6 — Frontend

---

### CARD-16 · Streamlit Frontend

**Tujuan:** Buat chat interface sederhana yang menampilkan tanya-jawab dengan agent beserta trace reasoning.

**Dependencies:** CARD-15

**File:** `frontend/app.py`

**Steps:**
1. Layout dasar:
   - Header: "AI Purchase Assistant (Demo)"
   - Disclaimer banner: "Demo project — data sintetis, bukan data produksi"
   - Chat history (scroll)
   - Input box + tombol Send
2. Tampilkan setiap respons dengan:
   - Jawaban utama (teks dari `final_answer`)
   - Expandable section "Lihat reasoning step-by-step" (tampilkan intent, data yang diambil, SHAP factors)
   - Badge waktu respons
3. Contoh pertanyaan (klik langsung ke input):
   - "Stok P-001 tinggal berapa?"
   - "Berapa unit P-003 yang perlu dibeli?"
   - "Kenapa penjualan P-007 turun?"
   - "Produk mana paling berisiko stockout?"
4. Tambahkan session state untuk riwayat chat dalam satu sesi
5. Koneksi ke FastAPI via `requests` (bukan langsung ke LangGraph)

**Output:** Chat UI berjalan di `localhost:8501`, demonstrasi penuh bisa dilakukan.

---

## FASE 7 — Observability & Evaluasi

---

### CARD-17 · LangSmith Integration

**Tujuan:** Pasang tracing LangSmith agar setiap eksekusi graph bisa dilihat step-by-step.

**Dependencies:** CARD-14

**Steps:**
1. Set environment variables:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=...
   LANGSMITH_PROJECT=procurement-agent
   ```
2. Verifikasi trace muncul di LangSmith dashboard untuk setiap graph run
3. Tambahkan custom metadata ke setiap trace: `intent`, `product_id`, `response_time_ms`
4. Buat satu "demo run" yang tracenya bisa dilink di portfolio (set public sharing di LangSmith)

**Output:** Trace LangSmith aktif, link trace publik siap untuk portfolio.

---

### CARD-18 · Backtesting & Evaluasi Model

**Tujuan:** Ukur akurasi forecasting secara formal menggunakan backtesting, dokumentasikan hasilnya.

**Dependencies:** CARD-05

**File:** `ml/evaluate.py`

**Steps:**
1. Backtest strategy: **walk-forward validation**
   - Train pada data s/d bulan N, test pada bulan N+1
   - Geser window, ulangi 3x
2. Hitung MAPE per kategori dan overall
3. Buat laporan singkat `ml/evaluation_report.md`:
   - MAPE per kategori
   - Contoh prediksi vs aktual (3-5 produk)
   - Feature importance chart (dari SHAP mean absolute value)
4. Jika MAPE > 20%: tuning parameter (lebih banyak estimators, feature tambahan)

**Output:** Laporan evaluasi terdokumentasi, MAPE ≤ 20%.

---

## FASE 8 — Deployment

---

### CARD-19 · Deploy ke Cloud (Demo Publik)

**Tujuan:** Deploy seluruh sistem ke cloud gratis untuk demo portofolio publik.

**Dependencies:** Semua card sebelumnya

**Steps:**

**Database:**
1. Buat project di Neon atau Supabase (PostgreSQL free tier)
2. Jalankan migration script (CARD-01) di cloud DB
3. Jalankan ETL (CARD-02) untuk load data ke cloud DB

**Backend:**
1. Buat `Dockerfile` untuk FastAPI:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Deploy ke Railway atau Render (free tier)
3. Set environment variables di dashboard: `DATABASE_URL`, `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`

**Frontend:**
1. Deploy Streamlit ke Streamlit Community Cloud
2. Update `API_URL` di frontend ke URL Railway/Render
3. Set secrets di Streamlit Cloud settings

**Security checklist sebelum go-live:**
- [ ] Rate limiting aktif (CARD-15)
- [ ] API key tidak ter-expose di frontend
- [ ] Disclaimer "demo project" tampil di UI
- [ ] SQL tools whitelist berjalan (CARD-03)
- [ ] Data yang dipakai 100% sintetis/publik

**Output:** App bisa diakses publik via URL, siap dimasukkan ke link portfolio.

---

## Summary — Urutan Pengerjaan

```
CARD-00  → Setup project
CARD-01  → Database schema
CARD-02  → ETL pipeline
CARD-03  → SQL tools
CARD-04  → Feature engineering
CARD-05  → Train XGBoost
CARD-06  → SHAP explainer
CARD-07  → LangGraph state + skeleton
CARD-08  → Router node
CARD-09  → Fetch data node
CARD-10  → Forecast node
CARD-11  → Explain node
CARD-12  → Decision node
CARD-13  → Polish node
CARD-14  → E2E graph test
CARD-15  → FastAPI backend
CARD-16  → Streamlit frontend
CARD-17  → LangSmith tracing
CARD-18  → Backtesting evaluasi
CARD-19  → Deploy publik
```

**Estimasi waktu (solo):** 2–3 minggu kerja aktif untuk MVP Fase 1 + Fase 2.
