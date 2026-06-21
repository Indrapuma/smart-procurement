# PRD: Agentic AI Purchase Decision Assistant

**Versi:** 1.0
**Penulis:** Indra Putra Mahayuda
**Tanggal:** Juni 2026
**Status:** Draft — Portofolio Project

---

## 1. Latar Belakang

Pengambilan keputusan pembelian barang (berapa unit, kapan reorder) di banyak bisnis ritel/distribusi masih dilakukan manual atau pakai aturan statis (mis. reorder point tetap), tanpa mempertimbangkan tren, musiman, atau faktor lain secara eksplisit. Selain itu, keputusan berbasis model ML (forecasting) sering dianggap "black box" sehingga sulit dipercaya oleh non-teknis.

Proyek ini membangun asisten AI agentic yang:
- Mengambil data operasional secara mandiri (lewat SQL tools)
- Menghasilkan prediksi demand dan rekomendasi qty pembelian
- Menjelaskan **alasan** di balik rekomendasi (XAI) dalam bahasa yang mudah dipahami oleh user non-teknis

## 2. Tujuan

| Tujuan | Deskripsi |
|---|---|
| Tujuan utama | Asisten dapat menjawab pertanyaan operasional inventory dengan reasoning multi-step yang transparan |
| Tujuan portofolio | Menunjukkan kemampuan membangun agentic AI (LangGraph) yang mengintegrasikan SQL, model ML, dan XAI secara terstruktur |
| Tujuan teknis | Membuktikan pemahaman kapan memakai tool terstruktur (SQL) vs reasoning kompleks (forecast+SHAP) vs RAG |

## 3. Target Pengguna (Persona)

- **Staf gudang/inventory** — butuh jawaban cepat soal stok saat ini
- **Manajer pembelian (non-teknis)** — butuh rekomendasi qty + alasan yang mudah dimengerti, tanpa istilah ML
- **Analis/Recruiter (untuk portofolio)** — ingin melihat reasoning step-by-step dan kualitas eksplainability

## 4. Ruang Lingkup (Scope)

### In-scope (MVP)
1. Cek stok saat ini (simple lookup)
2. Rekomendasi qty pembelian + penjelasan XAI
3. Penjelasan tren/penurunan-kenaikan penjualan (why-question)
4. Ranking produk berisiko stockout/overstock
5. Routing intent otomatis (agent pilih jalur pendek vs panjang)

### Out-of-scope (MVP)
- Eksekusi otomatis PO ke supplier (human-in-the-loop dulu)
- What-if scenario (simulasi harga/promo) — fase berikutnya
- RAG untuk SOP/kebijakan supplier — fase berikutnya (opsional)
- Multi-warehouse/multi-lokasi optimization

## 5. User Stories

| ID | Sebagai | Saya ingin | Sehingga |
|---|---|---|---|
| US-1 | Staf gudang | Tanya stok produk tertentu | Tahu sisa stok real-time tanpa buka sistem ERP |
| US-2 | Manajer pembelian | Tahu berapa unit yang perlu dibeli | Bisa ambil keputusan reorder tepat waktu |
| US-3 | Manajer pembelian | Tahu alasan di balik rekomendasi | Bisa percaya dan validasi keputusan AI |
| US-4 | Manajer pembelian | Tahu kenapa penjualan produk turun/naik | Bisa ambil tindakan korektif |
| US-5 | Manajer pembelian | Lihat produk mana paling urgent ditangani | Bisa prioritaskan waktu dan anggaran |

## 6. Arsitektur Solusi

### 6.1 High-level Flow
```
User Query
   ↓
Router Node (klasifikasi intent)
   ↓
┌─────────────┬──────────────────┬─────────────────┐
│ simple_lookup│ purchase_rec     │ explain_trend    │
↓             ↓                  ↓
fetch_data    fetch_data         fetch_data
   ↓             ↓ forecast          ↓ explain (SHAP)
   END           ↓ explain (SHAP)    ↓ polish
                 ↓ decision          ↓
                 ↓ polish            END
                 END
```

### 6.2 LangGraph State Schema
```python
class AgentState(TypedDict):
    user_query: str
    intent: str                # simple_lookup | purchase_recommendation | explain_trend | risk_ranking
    product_id: str
    sales_data: dict
    stock_data: dict
    forecast: dict
    shap_values: dict
    explanation_draft: str     # hasil rule-based template
    recommendation: dict       # qty, reorder point, dll
    final_answer: str          # hasil polish LLM
```

### 6.3 Node Definitions

| Node | Fungsi |
|---|---|
| `router_node` | Klasifikasi intent dari query user (LLM-based classification) |
| `fetch_data_node` | Panggil SQL tools sesuai kebutuhan intent |
| `forecast_node` | Jalankan model XGBoost untuk prediksi demand |
| `explain_node` | Hitung SHAP values + generate draft penjelasan rule-based |
| `decision_node` | Hitung rekomendasi qty (reorder point/EOQ) atau ranking risiko |
| `polish_node` | LLM merangkai draft + angka jadi jawaban final yang natural |

### 6.4 SQL Tools (Predefined, Read-only)

| Tool | Parameter | Output |
|---|---|---|
| `get_current_stock` | product_id | qty stok saat ini, lokasi |
| `get_sales_history` | product_id, n_months | data penjualan historis |
| `get_supplier_leadtime` | product_id | lead time supplier (hari) |
| `get_demand_trend` | product_id, n_months | growth rate, moving average |
| `get_seasonality_index` | product_id | index musiman |
| `rank_products_by_risk` | category (opsional) | list produk diurutkan berdasarkan urgency |

Semua query parameterized, whitelist tabel (`sales`, `inventory`, `products`, `suppliers`), tidak ada DML (insert/update/delete).

### 6.5 Model & XAI Layer

- **Model**: XGBoost Regressor, per kategori produk (atau per produk jika volume data cukup)
- **Explainability**: SHAP (TreeExplainer) untuk feature attribution per prediksi
- **Translation SHAP → bahasa**:
  - Tahap 1: rule-based template per fitur (trend, seasonality, price, leadtime, promo)
  - Tahap 2: LLM merangkai template jadi narasi yang mengalir dan kontekstual

### 6.6 Decision Logic (Reorder Quantity)

```
Reorder Qty = Forecasted Demand (lead time period) + Safety Stock − Current Stock

Safety Stock = Z-score (service level) × std_dev(demand) × sqrt(lead_time)
```

## 7. Tech Stack

| Layer | Teknologi |
|---|---|
| Orchestration | LangGraph |
| Backend API | FastAPI |
| Database | PostgreSQL |
| ML Model | XGBoost |
| Explainability | SHAP |
| LLM | Gemini (via Google AI Studio) |
| Frontend | Streamlit / React sederhana |
| Observability | LangSmith (trace graph execution) |

## 8. Skema Data (Ringkas)

```sql
products (product_id, name, category, unit_price)
sales (sale_id, product_id, date, qty_sold)
inventory (product_id, current_stock, last_updated)
suppliers (supplier_id, product_id, lead_time_days, min_order_qty)
```

## 9. Sumber Data (Dataset Publik)

Untuk kebutuhan demo dan training model, proyek ini menggunakan dataset publik (bukan data production riil), sehingga aman untuk dipakai di portofolio yang di-deploy publik.

### 9.1 Dataset Utama
**Retail Store Inventory Forecasting Dataset (Kaggle)**
- Link: https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
- ~73.000 baris data harian, multi-store dan multi-produk
- Kolom tersedia: `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Demand Forecast`, `Weather Condition`, `Holiday/Promotion`, `Price`
- Dipilih sebagai sumber utama karena fitur kontekstualnya (promo, cuaca, hari libur) cocok untuk showcase SHAP — bisa jelaskan kontribusi tiap faktor terhadap demand, bukan cuma tren waktu

### 9.2 Dataset Alternatif/Pelengkap
| Dataset | Kegunaan |
|---|---|
| Store Item Demand Forecasting Challenge (Kaggle) | Data lebih bersih, opsi untuk testing pipeline awal tanpa banyak preprocessing |
| Retail Store Inventory and Demand Forecasting (Kaggle, atomicd) | Alternatif dataset sintetis untuk variasi skenario demo |

### 9.3 Transformasi Data ke Skema Internal
Dataset publik di atas tidak punya kolom `lead_time` dan `supplier`, sehingga perlu digenerate secara sintetis (mis. random lead time 3–21 hari per kategori produk) saat proses ETL ke skema `products`, `sales`, `inventory`, `suppliers` yang sudah didefinisikan di Section 8.

### 9.4 Catatan Privasi & Keamanan
- Dataset bersifat sintetis/publik, tidak mengandung data pelanggan/bisnis riil
- Aman digunakan untuk demo yang di-deploy publik (lihat Section 13 — Deployment)

## 10. Contoh Interaksi

**Simple lookup:**
```
User: "Stok Produk A tinggal berapa?"
Agent: "Stok Produk A saat ini: 35 unit (update: hari ini 14:00)"
```

**Purchase recommendation:**
```
User: "Berapa unit Produk A yang perlu dibeli bulan depan?"
Agent: "Rekomendasi beli: 120 unit.

Alasan utama:
- Tren penjualan 3 bulan terakhir naik 18%
- Musim liburan mendekat, menaikkan prediksi ±10%
- Stok saat ini 35 unit, lead time supplier 14 hari"
```

**Explain trend:**
```
User: "Kenapa penjualan Produk B turun bulan ini?"
Agent: "Penjualan Produk B turun 12% dibanding bulan lalu, terutama karena
penurunan permintaan musiman dan tidak ada promosi aktif bulan ini."
```

## 11. Non-Functional Requirements

| Aspek | Target |
|---|---|
| Response time (simple_lookup) | < 2 detik |
| Response time (full pipeline) | < 8 detik |
| Akurasi forecast | MAPE < 20% (baseline, evaluasi lanjut sesuai data) |
| Keamanan SQL | Hanya predefined query, tidak ada raw SQL dari LLM |
| Transparansi | Setiap rekomendasi disertai breakdown alasan (SHAP-based) |

## 12. Metrik Keberhasilan (untuk demo portofolio)

- Agent berhasil membedakan intent dan memilih jalur node yang tepat (≥90% akurasi routing pada test set pertanyaan)
- Forecast akurasi terukur lewat backtesting (MAPE)
- Penjelasan XAI dapat dipahami non-teknis (validasi kualitatif)
- Trace LangSmith menunjukkan reasoning step-by-step yang jelas

## 13. Deployment

Untuk kebutuhan demo portofolio, sistem perlu di-deploy publik dengan setup ringan (bukan production-grade), menggunakan data dari Section 9 (dataset publik/sintetis).

| Komponen | Opsi |
|---|---|
| Backend (FastAPI + LangGraph) | Railway / Render / Fly.io (free tier) |
| Database (PostgreSQL) | Supabase / Neon (free tier) |
| Frontend (Streamlit/React) | Streamlit Community Cloud / Vercel |
| LLM API | API key milik developer, dengan rate limiting publik |
| Observability | LangSmith untuk trace graph execution (opsional, bagus untuk demo) |

**Pertimbangan keamanan saat deploy publik:**
- Rate limiting per IP/session untuk mencegah biaya API membengkak
- Gunakan data sintetis/publik saja, jangan expose data bisnis riil
- Tambahkan disclaimer "demo project" pada UI

## 14. Roadmap

| Fase | Deliverable |
|---|---|
| Fase 1 (MVP) | SQL tools, model forecasting, SHAP explain, decision logic, graph dasar (3 intent) |
| Fase 2 | Risk ranking node, polish node (LLM narasi), frontend chat sederhana |
| Fase 3 (opsional) | RAG node untuk SOP/kebijakan, what-if scenario, dashboard visualisasi SHAP |

## 15. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Data historis tidak cukup untuk produk baru | Fallback ke rata-rata kategori (cold start) |
| Forecast tidak akurat untuk demand volatile | Tampilkan confidence interval, jangan beri angka mutlak tanpa konteks |
| LLM halusinasi saat polish jawaban | Pisahkan fakta (rule-based) dari narasi (LLM), validasi angka tidak diubah LLM |
| Routing intent salah klasifikasi | Tambahkan fallback: jika ambigu, agent tanya klarifikasi ke user |

## 16. Out of Scope / Future Work

- Eksekusi otomatis PO ke sistem ERP
- Multi-warehouse optimization
- Real-time streaming data (saat ini batch/periodic)
- Integrasi langsung dengan supplier API
