---
name: rag-architecture
description: Use when building RAG systems, choosing between context stuffing vs vector search, evaluating embedding models, designing retrieval pipelines for LLM applications, or scaling an existing vector database
---

# RAG 架構決策與實作模式

## 核心原則

**RAG 不是 ML 問題，是資訊架構 + 資料建模問題。**（Doug Turnbull）

Vector search 應該是 fallback，不是預設。先問：「這個查詢能不能用 SQL 解決？」

## 本專案的三種向量 DB 用途

| 用途 | 說明 | 對應策略 |
|------|------|----------|
| **客戶 ERP 資料洞察** | 整理客戶的 POS/ERP 訂單數據，AI 顧問從中找出經營盲點與成長機會 | Context Stuffing（資料量小、需高精確） |
| **訪談紀錄整理** | 研究專題的游擊訪談、問卷回覆等質性資料，語意搜尋特定主題 | Orama hybrid → 未來 pgvector（多格式、規模成長中） |
| **本地開發查詢** | Soking 在開發過程中查詢產品路線圖、課程排程、營運文件等 | Orama 本地 hybrid search |

## 策略選擇框架

| 條件 | 策略 | 理由 |
|------|------|------|
| 單客戶資料 < 100k tokens | **Context Stuffing** | 100% 精確，零檢索錯誤，無需維護向量 DB |
| 結構化資料（日期/價格/狀態） | **SQL-first + Vector fallback** | 數值比較和精確篩選 embedding 做不到 |
| 非結構化文件 < 5,000 chunks | **本地 Orama hybrid search** | 零部署、無網路延遲、JSON 序列化 |
| 文件 > 5,000 chunks 或多租戶 | **Supabase pgvector** | 結構化過濾 + 向量排序一個查詢搞定 |

## Pattern A：Context Stuffing（客戶 ERP 資料洞察）

**適用**：每個客戶 1-5 份文件，< 3,000 tokens。客戶上傳 POS 訂單數據後，AI 顧問直接引用數字分析。

將客戶資料直接注入 system prompt，不需要 retrieval pipeline：

```
客戶資料 (order_data_chunks + quick_reports)
  → buildClientDataContext() + formatReportContext()
  → 注入 system prompt
  → LLM 直接引用具體數字回答
```

**關鍵設計**：
- 有客戶資料時 `maxTokens = 1000`，無資料時 `500`（動態 token 上限）
- 額外附加 `DATA_AWARE_GUIDANCE` 指引 LLM 引用具體數字
- Gemini Flash 的 1M context window 足夠容納（遠低於上限）

**實作參考**：`api/report/demo-agent-chat.ts`、`data/demo-default-prompt.ts`

## Pattern B：Orama 本地向量 DB（本地開發查詢 + 訪談紀錄）

**適用**：Soking 本地開發查詢（產品路線圖、課程排程、營運文件）+ 研究專題訪談紀錄的語意搜尋。目前 < 3,000 chunks / 62MB。

**技術棧**：
- Orama v3 — JSON 序列化、in-process、免部署
- `Xenova/multilingual-e5-small` — 384d、繁中支援、Node.js ONNX
- Hybrid search：全文關鍵字 + 向量相似度（threshold 0.5）

**索引 pipeline**：
```
.md 文件 → 按 ## 切 chunk (≤ 2000 chars)
  → 批次 embedding (16/batch)
  → Orama DB → persist → .vector-db/orama.json
增量更新：MD5 manifest 追蹤 → 只處理 changed/added/removed
```

**Supabase 表序列化**：結構化欄位 serialize 為可讀文字 → embedding（犧牲精確查詢換取語意搜尋）

**Multi-query RAG**：`expandQueries()` 用 LLM 生成 2-3 個查詢變體 → 平行搜尋 → 去重取最高分

**擴展性警告**：
- 🟡 2,874 chunks (62MB)：可用
- 🔴 28,000 chunks (~620MB)：JSON parse 分鐘級、記憶體爆、384d 向量空間擁擠

**實作參考**：`lib/agent/rag.ts`、`lib/agent/embeddings.ts`、`scripts/build-index.ts`

## Pattern C：pgvector（規劃路徑，訪談紀錄規模化 + 多客戶 ERP）

**適用**：訪談紀錄 CSV/PDF 增長到 Orama 上限後的遷移目標；或客戶 ERP 資料量超過 Context Stuffing 上限時的多租戶方案。

**核心價值** — 結構化過濾 + 向量排序合一：
```sql
SELECT *, embedding <=> $query_vec AS distance
FROM documents
WHERE category = '課程活動' AND metadata->>'status' = 'open'
ORDER BY distance LIMIT 10;
```

**Query Understanding 層**（推廣 `student-query.ts` 模式）：
```
使用者問題 → LLM 意圖分類
  → 結構化意圖 (70%)：直接 SQL
  → 語意意圖 (20%)：pgvector similarity
  → 混合意圖 (10%)：SQL filter + vector rerank
```

**多租戶隔離**：`client_id` 欄位 + Supabase RLS policy

**實作參考**：`lib/agent/student-query.ts`（已有 LLM → SQL 的模式可推廣）

## Embedding 模型選擇

| 模型 | 維度 | 適用場景 | 注意事項 |
|------|------|----------|----------|
| `multilingual-e5-small` | 384 | 本地開發、< 5k chunks | E5 需加 `query:` prefix |
| `multilingual-e5-base` | 768 | 中規模、較好語意分離 | 本地跑較慢 |
| OpenAI / Voyage API | 1536+ | 生產環境、大規模 | 有成本、避免 ONNX/Alpine 問題 |

**E5 使用規範**：查詢時加 `query: {text}` prefix，索引時加 `passage: {text}` prefix。

**Alpine Linux 陷阱**：`onnxruntime-node` 需要 glibc 的 `ld-linux-x86-64.so.2`，Zeabur Alpine 沒有。用 dynamic import 保護，生產環境改用 API-based embedding。

## Common Mistakes

| 錯誤 | 正確做法 |
|------|----------|
| 結構化資料全走 embedding | 日期/價格/狀態用 SQL，語意查詢才走 vector |
| Query expansion = Query understanding | expansion 是同質擴展；understanding 是拆解意圖為結構化條件 |
| JSON 序列化向量 DB 當作可擴展方案 | > 100MB 就該遷移 pgvector |
| 小量客戶資料也建 RAG pipeline | < 100k tokens 直接 Context Stuffing，省去檢索錯誤 |
| 通用 embedding 處理特定領域 | 規模大時 384d 空間擁擠，需升級維度或 fine-tune |

## soking.cc 完整參考檔案

| 檔案 | 用途 |
|------|------|
| `產品路線圖/10-RAG架構評估.md` | 三場景完整評估（本文件的源頭） |
| `lib/agent/rag.ts` | Orama hybrid search |
| `lib/agent/embeddings.ts` | Embedding singleton + 批次 |
| `scripts/build-index.ts` | 全量/增量索引 |
| `scripts/vector-db/supabase-indexer.ts` | Supabase 表序列化 |
| `api/report/demo-agent-chat.ts` | Context Stuffing 實作 |
| `data/demo-default-prompt.ts` | 資料格式化函式 |
| `api/agent/chat.ts` | Multi-query RAG |
| `lib/agent/student-query.ts` | LLM → SQL 結構化查詢 |
| `docs/planning/向量資料庫使用方式總覽.md` | 索引操作指南 |
