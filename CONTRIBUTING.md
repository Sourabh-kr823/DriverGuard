# Contributing to DriverGuard

## Team Workflow

### Setup
```bash
git clone https://github.com/Sourabh-kr823/DriverGuard.git
cd DriverGuard
conda create -n driverguard python=3.10
conda activate driverguard
pip install -r requirements.txt
```

Download `models/shape_predictor_68_face_landmarks.dat` from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place in `models/`.

### Before making changes
```bash
git pull origin main        # always pull latest before working
```

### After making changes
```bash
git add .
git commit -m "type: short description of what changed"
git push origin main
```

### Commit message types
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — cleanup, gitignore, dependencies
- `test:` — test scripts

### Files NOT to commit
- `*.db` database files
- `*.log` log files
- `*.mp4` video files
- `*.dat` large model files
- `*.backup.pt` backup weights
- `__pycache__/` folders

These are all in `.gitignore` already.

## Running the System
```bash
conda activate driverguard
cd DriverGuard
python main.py --simulate --preview
```
Open `http://localhost:5000` for the dashboard.
