from modules.database.db_manager import DatabaseManager
import yaml, time

cfg = yaml.safe_load(open('config.yaml'))
db = DatabaseManager(cfg['database'])
db.start()

# Inject at ALL simulation waypoints so GPS definitely hits one
waypoints = [
    (12.9716, 77.5946),
    (12.9776, 77.6000),
    (12.9820, 77.6080),
    (12.9870, 77.6120),
]
for lat, lon in waypoints:
    db.log_road_event(
        class_name='pothole', confidence=0.91,
        severity='severe', area_px2=200000,
        lat=lat, lon=lon
    )
print(f'Injected {len(waypoints)} test hazards ✓')
time.sleep(1)
db.stop()