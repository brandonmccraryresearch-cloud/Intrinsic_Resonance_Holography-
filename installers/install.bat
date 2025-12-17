@echo off
REM ==============================================================================
REM Intrinsic Resonance Holography (IRH) v21.1 - Windows Installation Script
REM 
REM THEORETICAL FOUNDATION: Intrinsic_Resonance_Holography-v21.1.md §1.6
REM Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-
REM
REM This script installs IRH and all dependencies on Windows systems.
REM ==============================================================================

setlocal EnableDelayedExpansion

REM Configuration
set "REPO_URL=https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-.git"
set "INSTALL_DIR=%USERPROFILE%\IRH"
set "VENV_NAME=irh_venv"
set "PYTHON_MIN_VERSION=3.10"

REM Colors (limited support in cmd)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM ==============================================================================
REM Main Script
REM ==============================================================================

:main
cls
echo.
echo %BLUE%========================================================================%NC%
echo %BLUE%    Intrinsic Resonance Holography (IRH) v21.1 - Windows Installer     %NC%
echo %BLUE%                                                                        %NC%
echo %BLUE%  Unified Theory of Emergent Reality from Quantum-Informational Principles%NC%
echo %BLUE%========================================================================%NC%
echo.
echo Installation directory: %INSTALL_DIR%
echo.

REM Check prerequisites
call :check_prerequisites
if errorlevel 1 goto :error

REM Create installation directory
call :create_install_directory
if errorlevel 1 goto :error

REM Clone or update repository
call :clone_repository
if errorlevel 1 goto :error

REM Create virtual environment
call :create_virtual_environment
if errorlevel 1 goto :error

REM Install dependencies
call :install_dependencies
if errorlevel 1 goto :error

REM Verify installation
call :verify_installation
if errorlevel 1 goto :error

REM Create launcher scripts
call :create_launcher_scripts
if errorlevel 1 goto :error

REM Setup PATH
call :setup_path

echo.
echo %GREEN%========================================================================%NC%
echo %GREEN%                    Installation Complete!                              %NC%
echo %GREEN%========================================================================%NC%
echo.
echo Quick Start:
echo   1. Open a new Command Prompt (to refresh PATH)
echo   2. Run: irh test           ^(Run the test suite^)
echo   3. Run: irh notebook       ^(Launch Jupyter notebooks^)
echo   4. Run: irh shell          ^(Python shell with IRH^)
echo.
echo Documentation:
echo   - README: %INSTALL_DIR%\README.md
echo   - Theory: %INSTALL_DIR%\Intrinsic_Resonance_Holography-v21.1.md
echo   - Notebooks: %INSTALL_DIR%\notebooks\
echo.
echo Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-
echo.
pause
goto :eof

:error
echo.
echo %RED%Installation failed. Please check the error messages above.%NC%
pause
exit /b 1

REM ==============================================================================
REM Helper Functions
REM ==============================================================================

:check_prerequisites
echo [i] Checking prerequisites...

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo %RED%[X] Python is not installed or not in PATH.%NC%
    echo     Please install Python %PYTHON_MIN_VERSION%+ from https://python.org
    echo     Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

REM Get Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo [+] Python %PYTHON_VERSION% found

REM Check pip
python -m pip --version >nul 2>nul
if errorlevel 1 (
    echo %YELLOW%[!] pip not found. Installing...%NC%
    python -m ensurepip --default-pip
)
echo [+] pip is available

REM Check git
where git >nul 2>nul
if errorlevel 1 (
    echo %RED%[X] Git is not installed or not in PATH.%NC%
    echo     Please install Git from https://git-scm.com/download/win
    exit /b 1
)
echo [+] Git is available

exit /b 0

:create_install_directory
echo [i] Creating installation directory...

if exist "%INSTALL_DIR%" (
    echo %YELLOW%[!] Directory already exists. Updating existing installation...%NC%
) else (
    mkdir "%INSTALL_DIR%"
)
echo [+] Installation directory ready
exit /b 0

:clone_repository
echo [i] Cloning IRH repository...

if exist "%INSTALL_DIR%\.git" (
    echo [i] Repository exists. Pulling latest changes...
    cd /d "%INSTALL_DIR%"
    git pull origin main 2>nul || git pull origin master
) else (
    git clone "%REPO_URL%" "%INSTALL_DIR%"
    cd /d "%INSTALL_DIR%"
)

echo [+] Repository cloned/updated
exit /b 0

:create_virtual_environment
echo [i] Creating Python virtual environment...

cd /d "%INSTALL_DIR%"

if exist "%VENV_NAME%" (
    echo %YELLOW%[!] Virtual environment exists. Using existing one...%NC%
) else (
    python -m venv "%VENV_NAME%"
)

REM Activate and upgrade pip
call "%VENV_NAME%\Scripts\activate.bat"
python -m pip install --upgrade pip

echo [+] Virtual environment created and activated
exit /b 0

:install_dependencies
echo [i] Installing dependencies...

cd /d "%INSTALL_DIR%"
call "%VENV_NAME%\Scripts\activate.bat"

if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo %YELLOW%[!] requirements.txt not found. Installing core dependencies...%NC%
    pip install numpy scipy sympy matplotlib jupyter pytest
)

echo [+] Dependencies installed
exit /b 0

:verify_installation
echo [i] Verifying installation...

cd /d "%INSTALL_DIR%"
call "%VENV_NAME%\Scripts\activate.bat"

python -c "from src.primitives.quaternions import Quaternion; from src.rg_flow.fixed_points import find_fixed_point; fp = find_fixed_point(); print(f'[+] Cosmic Fixed Point: lambda* = {fp.lambda_star:.3f}'); print('[+] Installation verified!')"
if errorlevel 1 (
    echo %RED%[X] Installation verification failed.%NC%
    exit /b 1
)

echo [+] Installation verified successfully
exit /b 0

:create_launcher_scripts
echo [i] Creating launcher scripts...

cd /d "%INSTALL_DIR%"

REM Create irh.bat launcher
(
echo @echo off
echo setlocal
echo set "IRH_HOME=%INSTALL_DIR%"
echo call "%%IRH_HOME%%\%VENV_NAME%\Scripts\activate.bat"
echo set "PYTHONPATH=%%IRH_HOME%%;%%PYTHONPATH%%"
echo.
echo if "%%1"=="test" ^(
echo     pytest tests/ -v %%2 %%3 %%4 %%5
echo     goto :eof
echo ^)
echo if "%%1"=="notebook" ^(
echo     jupyter lab notebooks/
echo     goto :eof
echo ^)
echo if "%%1"=="verify" ^(
echo     python scripts/verify_theoretical_annotations.py
echo     goto :eof
echo ^)
echo if "%%1"=="compute" ^(
echo     python scripts/compute_all_observables.py
echo     goto :eof
echo ^)
echo if "%%1"=="shell" ^(
echo     python
echo     goto :eof
echo ^)
echo.
echo echo IRH v21.1 - Intrinsic Resonance Holography
echo echo.
echo echo Usage: irh ^<command^>
echo echo.
echo echo Commands:
echo echo   test       Run the test suite
echo echo   notebook   Launch Jupyter Lab with notebooks
echo echo   verify     Verify theoretical annotations
echo echo   compute    Compute all observables
echo echo   shell      Open Python shell with IRH loaded
echo echo.
echo echo Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-
) > irh.bat

echo [+] Launcher script created
exit /b 0

:setup_path
echo [i] Setting up PATH...

REM Check if already in PATH
echo %PATH% | findstr /I /C:"%INSTALL_DIR%" >nul
if %errorlevel%==0 (
    echo [i] IRH already in PATH
    exit /b 0
)

REM Add to user PATH
setx PATH "%PATH%;%INSTALL_DIR%" >nul 2>nul
if errorlevel 1 (
    echo %YELLOW%[!] Could not add to PATH automatically.%NC%
    echo     Please add the following to your PATH manually:
    echo     %INSTALL_DIR%
) else (
    echo [+] Added IRH to PATH
    echo %YELLOW%[!] Please open a new Command Prompt to use 'irh' command.%NC%
)
exit /b 0
