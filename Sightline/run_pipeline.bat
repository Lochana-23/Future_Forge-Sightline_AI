@echo off
rem Usage: run_pipeline.bat [start_s] [end_s]     (defaults: 60 120)
rem Runs tracking on a window, then builds candidate events. Activate the virtual environment first.
setlocal
set START=%1
set END=%2
if "%START%"=="" set START=60
if "%END%"=="" set END=120
if not exist data\video_006.avi (
  echo Put video_006.avi in the data folder first.
  exit /b 1
)
if not exist config\zones.json (
  echo Draw the crosswalk zones first: python tools\define_zones.py data\video_006.avi --frame 1800 --out config\zones.json
  exit /b 1
)
python tools\run_tracking.py data\video_006.avi --start %START% --end %END% --stride 3
if errorlevel 1 exit /b 1
python tools\make_candidates.py data\video_006.avi
if errorlevel 1 exit /b 1
echo.
echo Done. Review the candidates with:  streamlit run app.py   (tab "Review candidates")
endlocal
