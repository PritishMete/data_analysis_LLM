# memory.py
import sqlite3
import json
import os
from datetime import datetime

# Use /tmp on Render (always writable)
DB_PATH = "/tmp/ai_memory.db"


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS command_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                raw_text    TEXT    NOT NULL,
                intent      TEXT    NOT NULL,
                slots       TEXT    NOT NULL,
                result      TEXT,
                success     INTEGER NOT NULL DEFAULT 0,
                confidence  REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS column_aliases (
                alias       TEXT PRIMARY KEY,
                real_name   TEXT NOT NULL,
                created_at  TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                original_text   TEXT    NOT NULL,
                wrong_intent    TEXT    NOT NULL,
                correct_intent  TEXT    NOT NULL,
                slots           TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[memory] init_db error: {e}")


def log_command(raw_text, intent, slots, result, success, confidence=0.0):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO command_log
               (timestamp, raw_text, intent, slots, result, success, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), raw_text, intent,
             json.dumps(slots), json.dumps(result), int(success), confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[memory] log_command error: {e}")


def get_recent_commands(limit=30):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """SELECT timestamp, raw_text, intent, slots, success, confidence
               FROM command_log ORDER BY id DESC LIMIT ?""", (limit,)
        ).fetchall()
        conn.close()
        return [{"time": r[0], "text": r[1], "intent": r[2],
                 "slots": json.loads(r[3]), "success": bool(r[4]),
                 "confidence": r[5]} for r in rows]
    except Exception as e:
        print(f"[memory] get_recent_commands error: {e}")
        return []


def get_command_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        total   = conn.execute("SELECT COUNT(*) FROM command_log").fetchone()[0]
        success = conn.execute("SELECT COUNT(*) FROM command_log WHERE success=1").fetchone()[0]
        by_intent = conn.execute(
            "SELECT intent, COUNT(*) FROM command_log GROUP BY intent ORDER BY COUNT(*) DESC"
        ).fetchall()
        conn.close()
        return {
            "total_commands": total,
            "successful":     success,
            "success_rate":   round(success / total * 100, 1) if total else 0,
            "by_intent":      [{"intent": r[0], "count": r[1]} for r in by_intent],
        }
    except Exception as e:
        print(f"[memory] get_command_stats error: {e}")
        return {"total_commands": 0, "successful": 0, "success_rate": 0, "by_intent": []}


def save_alias(alias, real_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO column_aliases (alias, real_name, created_at) VALUES (?,?,?)",
            (alias.lower().strip(), real_name.strip(), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[memory] save_alias error: {e}")


def resolve_alias(name):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT real_name FROM column_aliases WHERE alias=?",
            (name.lower().strip(),)
        ).fetchone()
        conn.close()
        return row[0] if row else name
    except Exception as e:
        print(f"[memory] resolve_alias error: {e}")
        return name


def get_all_aliases():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT alias, real_name, created_at FROM column_aliases"
        ).fetchall()
        conn.close()
        return [{"alias": r[0], "real_name": r[1], "created_at": r[2]} for r in rows]
    except Exception as e:
        print(f"[memory] get_all_aliases error: {e}")
        return []


def log_correction(original_text, wrong_intent, correct_intent, slots=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO corrections
               (timestamp, original_text, wrong_intent, correct_intent, slots)
               VALUES (?,?,?,?,?)""",
            (datetime.utcnow().isoformat(), original_text, wrong_intent,
             correct_intent, json.dumps(slots or {}))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[memory] log_correction error: {e}")


# Init on import — safe, uses /tmp
init_db()
