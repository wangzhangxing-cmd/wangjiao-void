@echo off
rem ------------------------------------------------------------------
rem  Run the v5 validation models one after another, then extract.
rem  Put this file next to build_fem_model.py, extract_fem_results.py and
rem  continue_job.py; the .inp and _meta.json files must be in "models".
rem  Start it from Task Scheduler ("Run whether user is logged on or not").
rem
rem  If Abaqus is terminated from outside before a job completes, the job
rem  is continued automatically from its last converged increment
rem  (restart data written by the model), up to MAXRESTARTS times.
rem  Progress is written to models\run_log.txt.
rem ------------------------------------------------------------------
setlocal
set CPUS=8
set MAXRESTARTS=10
rem V5_B00_EP_medium and V5_B10_EP_coarse are already done.
set JOBS=V5_B10_EP_medium
set HERE=%~dp0

cd /d "%HERE%models"
echo ==== start %date% %time% ==== >> run_log.txt
for %%J in (%JOBS%) do call :runjob %%J
echo ---- extraction started %date% %time% >> run_log.txt
call abaqus python "%HERE%extract_fem_results.py" >> run_log.txt 2>&1
echo ==== all done %date% %time% ==== >> run_log.txt
endlocal
goto :eof

:runjob
set BASE=%1
set CUR=%1
set COUNT=0
echo ---- %BASE% started %date% %time% >> run_log.txt
call abaqus job=%BASE% input=%BASE%.inp cpus=%CPUS% ask_delete=OFF interactive >> run_log.txt 2>&1
:check
findstr /C:"COMPLETED SUCCESSFULLY" "%CUR%.sta" >nul 2>&1
if not errorlevel 1 goto finished
if %COUNT% GEQ %MAXRESTARTS% goto gaveup
if exist next_job.txt del next_job.txt
call abaqus python "%HERE%continue_job.py" %BASE% >> run_log.txt 2>&1
if not exist next_job.txt goto gaveup
set NEXT=
set OLD=
for /f "usebackq tokens=1,2" %%a in ("next_job.txt") do (
    set NEXT=%%a
    set OLD=%%b
)
if "%NEXT%"=="" goto gaveup
if "%NEXT%"=="DONE" goto finished
set /a COUNT+=1
echo ---- %BASE%: restart %NEXT% from %OLD% (%COUNT%) %date% %time% >> run_log.txt
call abaqus job=%NEXT% oldjob=%OLD% cpus=%CPUS% ask_delete=OFF interactive >> run_log.txt 2>&1
set CUR=%NEXT%
goto check

:finished
echo ---- %BASE% finished %date% %time% >> run_log.txt
goto :eof

:gaveup
echo ---- %BASE% NOT completed, see %CUR%.msg %date% %time% >> run_log.txt
goto :eof
