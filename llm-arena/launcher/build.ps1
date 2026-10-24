param([switch]$Verify, [string]$OutputPath)

$ErrorActionPreference = 'Stop'
$arenaRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'The Windows .NET Framework C# compiler is required.' }
$outputExe = if ($OutputPath) { [IO.Path]::GetFullPath($OutputPath) } else { Join-Path $arenaRoot 'OpenRA-LLM-Arena.exe' }
$sources = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.cs' | Select-Object -ExpandProperty FullName)
& $compiler /nologo /target:winexe /platform:x64 /optimize+ /warnaserror+ "/out:$outputExe" /reference:System.Windows.Forms.dll /reference:System.Drawing.dll @sources
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
Write-Output ('Built ' + $outputExe)
if ($Verify) {
	$testExe = Join-Path $arenaRoot 'llm-arena/.local/launcher-argument-tests.exe'
	New-Item -ItemType Directory -Path (Split-Path -Parent $testExe) -Force | Out-Null
	& $compiler /nologo /target:exe /platform:x64 /warnaserror+ "/out:$testExe" (Join-Path $PSScriptRoot 'LaunchPaths.cs') (Join-Path $arenaRoot 'llm-arena/tests/LauncherArgumentTests.cs')
	if ($LASTEXITCODE -ne 0) { throw 'Launcher argument test compilation failed.' }
	& $testExe
	if ($LASTEXITCODE -ne 0) { throw 'Launcher argument regression tests failed.' }
	$restartTestExe = Join-Path $arenaRoot 'llm-arena/.local/launcher-restart-tests.exe'
	& $compiler /nologo /target:exe /platform:x64 /warnaserror+ "/out:$restartTestExe" (Join-Path $PSScriptRoot 'LaunchPaths.cs') (Join-Path $PSScriptRoot 'LauncherSession.cs') (Join-Path $arenaRoot 'llm-arena/tests/LauncherRestartTests.cs')
	if ($LASTEXITCODE -ne 0) { throw 'Launcher restart test compilation failed.' }
	& $restartTestExe
	if ($LASTEXITCODE -ne 0) { throw 'Launcher restart regression tests failed.' }
}
