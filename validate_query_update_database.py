import sqlite3
DB_PATH = "chat_history.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    SELECT *
    FROM chat_history
    ORDER BY id DESC
""")
rows = cursor.fetchall()
conn.close()

# Print all rows
for row in rows:
    print(row)
