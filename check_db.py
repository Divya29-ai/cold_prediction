import sqlite3

# connect to database
conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

# get all rows
cursor.execute("SELECT * FROM predictions")
rows = cursor.fetchall()

# print rows
for row in rows:
    print(row)

conn.close()