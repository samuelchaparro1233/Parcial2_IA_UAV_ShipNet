@echo off
echo =====================================================================
echo  INICIANDO SISTEMA DE PERCEPCION MARITIMA UAV (EVALUACION ABET)
echo =====================================================================
echo.
if exist "C:\Python314\python.exe" (
    "C:\Python314\python.exe" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
pause
