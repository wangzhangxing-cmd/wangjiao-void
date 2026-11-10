@echo off
rem ------------------------------------------------------------------
rem  Run the v5 validation models one after another, then extract.
rem  Put this file next to build_fem_model.py / extract_fem_results.py;
rem  the .inp and _meta.json files must be in the "models" subfolder.
rem  Start it from Task Scheduler ("Run whether user is logged on or not")
rem  so the jobs survive Remote Desktop disconnects and CAE crashes.
rem  Edit JOBS and CPUS below.  Progress is written to models\run_log.txt.
rem ------------------------------------------------------------------
setlocal
set CPUS=8
set JOBS=V5_B00_EP_medium V5_B10_EP_medium V5_B10_EP_coarse

cd /d "%~dp0models"
echo ==== start %date% %time% ==== >> run_log.txt
for %%J in (%JOBS%) do (
    echo ---- %%J started %date% %time% >> run_log.txt
    call abaqus job=%%J input=%%J.inp cpus=%CPUS% ask_delete=OFF interactive >> run_log.txt 2>&1
    echo ---- %%J finished %date% %time% >> run_log.txt
)
echo ---- extraction started %date% %time% >> run_log.txt
call abaqus python "%~dp0extract_fem_results.py" >> run_log.txt 2>&1
echo ==== all done %date% %time% ==== >> run_log.txt
endlocal
