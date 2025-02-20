@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "livehtml" goto livehtml
if "%1" == "md2rst" goto md2rst
if "%1" == "preview" goto preview
if "%1" == "todos" goto todos
if "%1" == "check" goto check
if "%1" == "full" goto full
if "%1" == "dev" goto dev

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:livehtml
sphinx-autobuild %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
goto end

:md2rst
echo Converting Markdown files to RST...
for /r %%i in (*.md) do (
    if not "%%i"=="%BUILDDIR%\*.md" (
        pandoc "%%i" -f markdown -t rst -o "%%~dpni.rst"
    )
)
goto end

:preview
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
start "" "%BUILDDIR%\html\index.html"
goto end

:todos
%SPHINXBUILD% -M todo %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:check
call :clean
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%SPHINXBUILD% -M linkcheck %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%SPHINXBUILD% -M doctest %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:full
call :clean
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%SPHINXBUILD% -M latexpdf %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%SPHINXBUILD% -M epub %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:dev
call :clean
goto livehtml

:end
popd