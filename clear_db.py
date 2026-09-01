import sqlite3

conn = sqlite3.connect('data/logs/driver_guard_events.db')
conn.execute('DELETE FROM road_events')
conn.commit()
conn.close()
print('Road events cleared ✓')