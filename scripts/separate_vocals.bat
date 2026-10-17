@echo off
rem Split songs into vocals and instrumental with a UVR Roformer model.
rem Usage: scripts\separate_vocals.bat <audio file or folder> [output folder]
setlocal

set "ROOT=%~dp0.."
set "SEPARATOR=%ROOT%\separator\env\Scripts\audio-separator.exe"
set "MODEL=vocals_mel_band_roformer.ckpt"

if not exist "%SEPARATOR%" (
    echo audio-separator not found: %SEPARATOR%
    exit /b 1
)
where ffmpeg >nul 2>nul || (
    echo ffmpeg not found on PATH. Open a new terminal after installing ffmpeg.
    exit /b 1
)
if "%~1"=="" (
    echo Usage: %~nx0 ^<audio file or folder^> [output folder]
    exit /b 1
)

set "OUTPUT=%~2"
if "%OUTPUT%"=="" set "OUTPUT=%ROOT%\separated"

"%SEPARATOR%" "%~1" -m %MODEL% --model_file_dir "%ROOT%\separator\models" --output_dir "%OUTPUT%" --output_format WAV
