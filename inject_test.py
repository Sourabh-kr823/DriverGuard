from modules.database.db_manager import DatabaseManager
import yaml, time

cfg = yaml.safe_load(open('config.yaml'))
db = DatabaseManager(cfg['database'])
db.start()
db.log_road_event(
    class_name='pothole', confidence=0.91,
    severity='severe', area_px2=200000,
    lat=12.9776, lon=77.6000
)
print('Test hazard injected ✓')
time.sleep(1)
db.stop()