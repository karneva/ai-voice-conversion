@echo off
rem Split songs into vocals and instrumental with a UVR Roformer model.
rem Usage: scripts\separate_vocals.bat <audio file or folder> [output folder]
rem Compressed inputs (m4a, mp3, ...) are decoded to WAV first; see separate_vocals.py.
"%~dp0..\separator\env\python.exe" "%~dp0separate_vocals.py" %*
