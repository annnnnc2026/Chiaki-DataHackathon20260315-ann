"""
虎菇婆餐廳向量資料庫建置腳本
模型：intfloat/multilingual-e5-small（384 維）
向量 DB：ChromaDB（本地持久化）
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import openpyxl
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# 路徑設定
# ============================================================

ROOT = Path("/Users/annchiang/Chiaki-DataHackathon20260315-ann")
DATA_DIR = ROOT / "練習資料" / "resraurant-Tiger"
ORDER_DIR = DATA_DIR / "虎菇婆_訂單"
WEATHER_DIR = DATA_DIR / "氣象資料"
VECTOR_DB_DIR = ROOT / ".vector-db" / "tiger-chroma"

TEMP_CSV = WEATHER_DIR / "板橋逐時氣溫.csv"
RAIN_CSV = WEATHER_DIR / "永和逐時降雨.csv"

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "tiger_orders"
BATCH_SIZE = 32

DAY_NAMES = ["週日", "週一", "週二", "週三", "週四", "週五", "週六"]


# ============================================================
# Step 1：載入氣象資料
# ============================================================

def load_weather() -> dict:
    """
    回傳 dict，key = "YYYY-MM-DD HH"（24hr，補零），
    value = {"temperature": float, "rainfall": float}
    """
    weather = {}

    # 板橋氣溫：格式 "2025-11-24 01:00" 或帶 BOM
    temp_df = pd.read_csv(TEMP_CSV, encoding="utf-8-sig")
    temp_col = temp_df.columns[0]   # 日期時間欄
    for _, row in temp_df.iterrows():
        raw = str(row[temp_col]).strip()
        # 取前 13 碼 "YYYY-MM-DD HH"
        key = raw[:13]
        try:
            weather[key] = {"temperature": float(row.iloc[1]), "rainfall": 0.0}
        except (ValueError, TypeError):
            pass

    # 永和降雨：格式 "2025-11-24T00:00"（ISO），補 temperature 若已有
    rain_df = pd.read_csv(RAIN_CSV, encoding="utf-8-sig")
    rain_col = rain_df.columns[0]
    for _, row in rain_df.iterrows():
        raw = str(row[rain_col]).strip()
        # normalize "T" → " "，取前 13 碼
        key = raw[:13].replace("T", " ")
        try:
            rainfall = float(row.iloc[1])
        except (ValueError, TypeError):
            rainfall = 0.0
        if key in weather:
            weather[key]["rainfall"] = rainfall
        else:
            weather[key] = {"temperature": 0.0, "rainfall": rainfall}

    print(f"[氣象] 載入 {len(weather)} 筆逐小時資料")
    return weather


# ============================================================
# Step 2：解析訂單 Excel
# ============================================================

def parse_order_time(raw) -> Optional[datetime]:
    """解析各種格式的下單時間"""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    raw_str = str(raw).strip()
    # 常見格式：2025/11/24 11:23:08AM 或 2025/11/24 11:23:08
    raw_str = re.sub(r'(\d{4})/(\d{2})/(\d{2})', r'\1-\2-\3', raw_str)
    for fmt in ("%Y-%m-%d %I:%M:%S%p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw_str, fmt)
        except ValueError:
            continue
    return None


def read_all_orders() -> list:
    """讀取所有 xlsx，回傳 list of dict（每筆訂單）"""
    xlsx_files = sorted(ORDER_DIR.glob("*.xlsx"))
    all_orders = []

    for xlsx_path in xlsx_files:
        print(f"  讀取：{xlsx_path.name}")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.active

        headers = None
        for row in ws.iter_rows(values_only=True):
            if headers is None:
                headers = [str(h).strip() if h else "" for h in row]
                continue
            if not any(row):
                continue
            record = dict(zip(headers, row))

            order_id = str(record.get("訂單ID", "")).strip()
            if not order_id or order_id == "None":
                continue

            order_time = parse_order_time(record.get("下單時間"))
            if not order_time:
                continue

            all_orders.append({
                "order_id": order_id,
                "store": str(record.get("店家", "虎菇婆")).strip(),
                "platform": str(record.get("點餐平台", "")).strip(),
                "dining_type": str(record.get("用餐型態", "")).strip(),
                "total_amount": float(record.get("總金額") or 0),
                "delivery_fee": float(record.get("運費") or 0),
                "discount": float(record.get("折扣") or 0),
                "order_time": order_time,
                "is_cancelled": record.get("是否已取消") in ("是", True, 1),
                "items": str(record.get("品項", "")).strip(),
                "table_no": str(record.get("桌號", "")).strip(),
                "delivery_address": str(record.get("外送地址", "")).strip(),
                "notes": str(record.get("備註", "")).strip(),
                "delivery_notes": str(record.get("外送訂單備註", "")).strip(),
            })
        wb.close()

    print(f"[訂單] 共讀取 {len(all_orders)} 筆")
    return all_orders


# ============================================================
# Step 3：序列化訂單文字（包含氣象）
# ============================================================

def get_hour_period(hour: int) -> str:
    if hour < 6:   return "凌晨"
    if hour < 11:  return "早上"
    if hour < 14:  return "中午"
    if hour < 18:  return "下午"
    if hour < 21:  return "晚上"
    return "深夜"


def serialize_order(order: dict, weather: dict) -> Tuple[str, dict]:
    """
    回傳 (content_text, metadata_dict)
    content_text 加 passage: 前綴（E5 規範）
    """
    dt: datetime = order["order_time"]
    date_str = f"{dt.year}年{dt.month}月{dt.day}日"
    day_str = DAY_NAMES[dt.weekday() + 1 if dt.weekday() < 6 else 0]
    # Python weekday(): 0=Mon … 6=Sun；轉為 0=Sun…6=Sat
    py_dow = dt.weekday()  # 0=Mon
    dow = (py_dow + 1) % 7   # 0=Sun,1=Mon,...,6=Sat

    time_str = f"{get_hour_period(dt.hour)}{dt.hour}點{dt.minute:02d}分"
    weather_key = f"{dt.strftime('%Y-%m-%d')} {dt.hour:02d}"
    w = weather.get(weather_key, {"temperature": 0.0, "rainfall": 0.0})

    parts = [
        f"訂單 {order['order_id']} | {order['store']} | {date_str}（{day_str}）{time_str}",
    ]
    if order["dining_type"]:
        parts.append(f"用餐方式：{order['dining_type']}")
    if order["platform"]:
        parts.append(f"點餐平台：{order['platform']}")
    if order["items"] and order["items"] not in ("None", ""):
        parts.append(f"品項：{order['items']}")
    parts.append(f"總金額：${int(order['total_amount'])}")
    if order["delivery_fee"] > 0:
        parts.append(f"外送費：${int(order['delivery_fee'])}")
    if order["discount"] > 0:
        parts.append(f"折扣：-${int(order['discount'])}")
    if order["table_no"] and order["table_no"] not in ("None", ""):
        parts.append(f"桌號：{order['table_no']}")
    if order["delivery_address"] and order["delivery_address"] not in ("None", ""):
        parts.append(f"外送地址：{order['delivery_address']}")
    if order["notes"] and order["notes"] not in ("None", ""):
        parts.append(f"備註：{order['notes']}")
    if order["delivery_notes"] and order["delivery_notes"] not in ("None", ""):
        parts.append(f"外送備註：{order['delivery_notes']}")

    temp = w["temperature"]
    rain = w["rainfall"]
    parts.append(f"當時氣溫：{temp}℃")
    if rain > 0:
        parts.append(f"降水量：{rain}mm（有雨）")
    else:
        parts.append("降水量：0mm（無雨）")

    content = " | ".join(parts)

    metadata = {
        "order_id": order["order_id"],
        "date": dt.strftime("%Y-%m-%d"),
        "hour": int(dt.hour),
        "day_of_week": int(dow),          # 0=Sun, 1=Mon, ..., 6=Sat
        "platform": order["platform"] or "未知",
        "dining_type": order["dining_type"] or "未知",
        "total_amount": int(order["total_amount"]),
        "is_cancelled": bool(order["is_cancelled"]),
        "temperature": float(round(temp, 1)),
        "rainfall": float(round(rain, 2)),
        "has_rain": bool(rain > 0),
    }

    return content, metadata


# ============================================================
# Step 4：建立 ChromaDB 並批次 embedding
# ============================================================

def build_index(orders: list, weather: dict):
    print(f"\n[向量DB] 初始化 ChromaDB：{VECTOR_DB_DIR}")
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # 刪除舊 collection（重建索引）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[向量DB] 刪除舊 collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    print(f"\n[模型] 載入 {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("[模型] 載入完成！")

    # 序列化所有訂單
    ids, documents, metadatas = [], [], []
    for order in orders:
        content, meta = serialize_order(order, weather)
        ids.append(order["order_id"])
        documents.append(content)
        metadatas.append(meta)

    # 處理重複 order_id（同 ID 多行的情況）
    seen = {}
    unique_ids, unique_docs, unique_metas = [], [], []
    for oid, doc, meta in zip(ids, documents, metadatas):
        if oid in seen:
            # 合併品項（在已有的 doc 後面附加新品項）
            idx = seen[oid]
            unique_docs[idx] += f"；{doc.split('品項：')[1].split(' | ')[0]}" if "品項：" in doc else ""
        else:
            seen[oid] = len(unique_ids)
            unique_ids.append(oid)
            unique_docs.append(doc)
            unique_metas.append(meta)

    total = len(unique_ids)
    print(f"\n[Embedding] 共 {total} 筆訂單，開始批次 embedding（batch={BATCH_SIZE}）...")

    all_embeddings = []
    for i in range(0, total, BATCH_SIZE):
        batch_docs = unique_docs[i:i + BATCH_SIZE]
        # E5 規範：索引時加 passage: 前綴
        prefixed = [f"passage: {doc}" for doc in batch_docs]
        embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        done = min(i + BATCH_SIZE, total)
        pct = done / total * 100
        print(f"  進度：{done}/{total}（{pct:.0f}%）", end="\r", flush=True)

    print(f"\n[Embedding] 完成！")

    # 分批插入 ChromaDB（每次最多 5461 筆，避免 SQLite 限制）
    CHROMA_BATCH = 500
    for i in range(0, total, CHROMA_BATCH):
        collection.add(
            ids=unique_ids[i:i + CHROMA_BATCH],
            documents=unique_docs[i:i + CHROMA_BATCH],
            embeddings=all_embeddings[i:i + CHROMA_BATCH],
            metadatas=unique_metas[i:i + CHROMA_BATCH],
        )

    print(f"[向量DB] 已插入 {collection.count()} 筆訂單到 ChromaDB")
    return collection


# ============================================================
# Step 5：驗證查詢
# ============================================================

def verify_query(collection, model: SentenceTransformer):
    print("\n[驗證] 語意查詢測試：「下雨天的外送訂單」")
    query_embedding = model.encode(
        ["query: 下雨天的外送訂單"],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        where={"has_rain": True},
    )

    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            dist = results["distances"][0][i]
            print(f"\n  [{i+1}] 相似度：{1-dist:.3f}")
            print(f"       {doc[:120]}...")
    else:
        print("  （無符合結果）")


# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 60)
    print("  虎菇婆餐廳向量資料庫建置")
    print(f"  模型：{MODEL_NAME}")
    print(f"  輸出：{VECTOR_DB_DIR}")
    print("=" * 60)

    # 1. 氣象
    print("\n[1/4] 載入氣象資料...")
    weather = load_weather()

    # 2. 訂單
    print("\n[2/4] 讀取訂單 Excel...")
    orders = read_all_orders()

    # 3. 建索引
    print("\n[3/4] 建立向量索引...")
    collection = build_index(orders, weather)

    # 4. 驗證
    print("\n[4/4] 驗證查詢...")
    model = SentenceTransformer(MODEL_NAME)
    verify_query(collection, model)

    print("\n" + "=" * 60)
    print("  ✅ 向量資料庫建置完成！")
    print(f"  📦 位置：{VECTOR_DB_DIR}")
    print(f"  📊 總筆數：{collection.count()} 筆訂單")
    print("=" * 60)


if __name__ == "__main__":
    main()
