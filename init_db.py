import sqlite3

conn = sqlite3.connect("users.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    preferred_domain_AI INTEGER,
    preferred_domain_ML INTEGER,
    preferred_domain_DS INTEGER,
    experience_years INTEGER
)
""")

conn.commit()
conn.close()

print("✅ predictions table ready")