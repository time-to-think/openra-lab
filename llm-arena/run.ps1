param(
    [switch]$CheckCloud,
    [switch]$CheckMcp,
    [switch]$ListModels,
    [string]$GlmModel,
    [string]$DeepSeekModel,
	[string]$Players,
    [Nullable[double]]$Interval,
    [string[]]$ArenaArgs = @()
)

$ErrorActionPreference = 'Stop'
$arenaPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $arenaPython)) { throw 'Run llm-arena/setup.ps1 first.' }

$previousKey = $env:OLLAMA_API_KEY
try {
    if (-not $CheckMcp -and -not $env:OLLAMA_API_KEY) {
        $arenaKeyFile = Join-Path $PSScriptRoot '.local/ollama-key.xml'
        if (Test-Path -LiteralPath $arenaKeyFile) {
            try { $secureKey = Import-Clixml -LiteralPath $arenaKeyFile }
            catch { throw 'Cannot decrypt the saved key. Run setup.ps1 -ReplaceApiKey under this Windows account.' }
            $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
            try { $env:OLLAMA_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer) }
            finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
                $secureKey.Dispose()
            }
        }
        else { $env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY', 'User') }
    }
    $options = @()
    if ($CheckCloud) { $options += '--check-cloud' }
    if ($CheckMcp) { $options += '--check-mcp' }
    if ($ListModels) { $options += '--list-models' }
    if ($GlmModel) { $options += @('--glm-model', $GlmModel) }
    if ($DeepSeekModel) { $options += @('--deepseek-model', $DeepSeekModel) }
	if ($Players) { $options += @('--players', $Players) }
    if ($null -ne $Interval) { $options += @('--interval', $Interval.ToString([Globalization.CultureInfo]::InvariantCulture)) }
    & $arenaPython (Join-Path $PSScriptRoot 'arena.py') @options @ArenaArgs
    $arenaExitCode = $LASTEXITCODE
}
finally { $env:OLLAMA_API_KEY = $previousKey }
exit $arenaExitCode
