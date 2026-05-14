import sqlite3

conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    location TEXT NOT NULL,
    status TEXT DEFAULT 'Pending'
)
""")

conn.commit()
conn.close()
print("Database initialized successfully.")