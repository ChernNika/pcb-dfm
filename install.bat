@echo off
chcp 65001 >nul
setlocal
rem Установка плагина «Проверка технологичности» для KiCad 10 (Windows).
rem Создаёт окружение Python, ставит пакет и копирует плагин в папку плагинов KiCad.
rem Повторный запуск обновляет установку (например, после git pull).
rem Окружение по умолчанию — %USERPROFILE%\.venvs\pcb-dfm (переменная DFM_VENV меняет путь).
rem Аргументы передаются в «python -m dfm plugin-install», например: install.bat --dest D:\plugins\ru.dfm.check

cd /d "%~dp0"
if not defined DFM_VENV set "DFM_VENV=%USERPROFILE%\.venvs\pcb-dfm"

set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PY=python"
if not defined PY (
    echo Не найден Python 3.11 или новее. Установите его с https://www.python.org/downloads/
    echo и запустите install.bat снова.
    goto fail
)

if not exist "%DFM_VENV%\Scripts\python.exe" (
    echo [1/3] Создаю окружение %DFM_VENV%
    %PY% -m venv "%DFM_VENV%" || goto fail
) else (
    echo [1/3] Окружение уже есть: %DFM_VENV%
)

echo [2/3] Ставлю пакет и зависимости (numpy, scipy, shapely, matplotlib), нужен интернет
"%DFM_VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -q . || goto fail

echo [3/3] Копирую плагин в KiCad
"%DFM_VENV%\Scripts\python.exe" -m dfm plugin-install %* || goto fail

echo.
echo Готово. Дальше в KiCad 10:
echo   1. Настройки - Параметры - Плагины: включить «Сервер IPC API», перезапустить KiCad.
echo   2. В редакторе плат нажать кнопку «Проверить технологичность»
echo      (при первом нажатии KiCad сам ставит kicad-python, нужен интернет).
if not defined DFM_NO_PAUSE pause
exit /b 0

:fail
echo.
echo Установка не завершена: см. сообщения выше.
if not defined DFM_NO_PAUSE pause
exit /b 1
