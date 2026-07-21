@echo off
REM Deploy dashboard updates to AWS (Windows wrapper)
REM Usage: deploy_dashboard.bat

echo === Deploying Dashboard to AWS ===
echo.

SET SERVER=ubuntu@54.172.65.25
SET KEY=trading-key-v2.pem
SET CONTAINER=algotrader

echo 1. Copying files to host...
scp -P 2222 -i "%KEY%" -r dashboard\frontend\dist "%SERVER%:~/algo-trading/dashboard/frontend/"

echo 2. Copying files into container...
ssh -p 2222 -i "%KEY%" "%SERVER%" "docker cp ~/algo-trading/dashboard/frontend/dist/index.html %CONTAINER%:/app/dashboard/frontend/dist/index.html"

REM Get bundle names from index.html
for /f "tokens=*" %%i in ('findstr /r "index-.*\.js" dashboard\frontend\dist\index.html') do (
    set LINE=%%i
    goto :found_js
)
:found_js
for /f "tokens=3 delims=/" %%a in ("%LINE%") do (
    for /f "tokens=1 delims=^"" %%b in ("%%a") do (
        set BUNDLE_JS=%%b
    )
)

for /f "tokens=*" %%i in ('findstr /r "index-.*\.css" dashboard\frontend\dist\index.html') do (
    set LINE=%%i
    goto :found_css
)
:found_css
for /f "tokens=3 delims=/" %%a in ("%LINE%") do (
    for /f "tokens=1 delims=^"" %%b in ("%%a") do (
        set BUNDLE_CSS=%%b
    )
)

echo    Copying %BUNDLE_JS%...
ssh -p 2222 -i "%KEY%" "%SERVER%" "docker cp ~/algo-trading/dashboard/frontend/dist/assets/%BUNDLE_JS% %CONTAINER%:/app/dashboard/frontend/dist/assets/"

echo    Copying %BUNDLE_CSS%...
ssh -p 2222 -i "%KEY%" "%SERVER%" "docker cp ~/algo-trading/dashboard/frontend/dist/assets/%BUNDLE_CSS% %CONTAINER%:/app/dashboard/frontend/dist/assets/"

echo.
echo Dashboard deployment complete!
echo.
echo Refresh your browser to see changes (Ctrl+R)
echo.
pause
