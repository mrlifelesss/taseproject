@echo off
REM ─────────────────────────────────────────────────────────────
REM Activate the Python virtual environment for this project
REM ─────────────────────────────────────────────────────────────

REM switch to the drive and folder
cd /d G:\tase_project

REM activate the env
call env\Scripts\activate.bat

REM keep the window open so you can see any messages
cmd /k
