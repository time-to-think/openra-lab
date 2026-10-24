param(
    [switch]$SkipApiKey,
    [switch]$ReplaceApiKey,
    [switch]$WithTraining
)

$ErrorActionPreference = 'Stop'
$arenaPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$arenaKeyFile = Join-Path $PSScriptRoot '.local/ollama-key.xml'

if (-not (Test-Path -LiteralPath $arenaPython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python 3.11 or later is required. Install Python, then run this script again.'
    }
    & $pythonCommand.Source -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
}

& $arenaPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($WithTraining) {
    & $arenaPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-rl.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
}
& $arenaPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python dependency conflicts found.' }
& $arenaPython (Join-Path $PSScriptRoot 'arena.py') --check-mcp
if ($LASTEXITCODE -ne 0) { throw 'MCP startup check failed.' }

if ($SkipApiKey) {
    Write-Host 'Local environment ready. Run setup.ps1 again to enter your Ollama Cloud key.'
    return
}

$existingKey = $env:OLLAMA_API_KEY
if (-not $existingKey) { $existingKey = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY', 'User') }
if ($ReplaceApiKey -or (-not $existingKey -and -not (Test-Path -LiteralPath $arenaKeyFile))) {
    Write-Host 'Create an API key at https://ollama.com/settings/keys. Input is hidden.'
    $secureKey = Read-Host 'Ollama Cloud API key' -AsSecureString
    if ($secureKey.Length -eq 0) { throw 'No key entered. Local dependencies are ready; rerun setup.ps1 to finish.' }
    New-Item -ItemType Directory -Path (Split-Path -Parent $arenaKeyFile) -Force | Out-Null
    # Windows DPAPI restricts decryption to this Windows user on this computer.
    $secureKey | Export-Clixml -LiteralPath $arenaKeyFile
    $secureKey.Dispose()
    Write-Host 'Key encrypted for this Windows account in the ignored .local directory.'
}

$previousKey = $env:OLLAMA_API_KEY
try {
    # A replacement must be tested even if the shell still holds the old key.
    if ($ReplaceApiKey) { $env:OLLAMA_API_KEY = $null }
    & (Join-Path $PSScriptRoot 'run.ps1') -CheckCloud
    if ($LASTEXITCODE -ne 0) { throw 'Cloud validation failed. Follow the error above and rerun setup.ps1.' }
}
finally { $env:OLLAMA_API_KEY = $previousKey }
