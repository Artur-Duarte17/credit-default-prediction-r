Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RscriptPath {
  $command = Get-Command Rscript.exe -ErrorAction SilentlyContinue
  if ($null -ne $command -and (Test-Path $command.Source)) {
    return $command.Source
  }

  $registryKeys = @(
    "HKLM:\SOFTWARE\R-core\R",
    "HKCU:\SOFTWARE\R-core\R",
    "HKLM:\SOFTWARE\WOW6432Node\R-core\R",
    "HKCU:\SOFTWARE\WOW6432Node\R-core\R"
  )

  $installPaths = foreach ($key in $registryKeys) {
    $item = Get-ItemProperty $key -ErrorAction SilentlyContinue
    if ($null -ne $item -and $item.PSObject.Properties.Name -contains "InstallPath") {
      $item.InstallPath
    }
  }

  $candidatePaths = foreach ($installPath in ($installPaths | Select-Object -Unique)) {
    Join-Path $installPath "bin\Rscript.exe"
    Join-Path $installPath "bin\x64\Rscript.exe"
  }

  foreach ($candidate in ($candidatePaths | Select-Object -Unique)) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  throw "Nao foi possivel localizar o Rscript.exe. Instale o R ou ajuste o PATH."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mainScript = Join-Path $projectRoot "main.R"

if (-not (Test-Path $mainScript)) {
  throw "Arquivo main.R nao encontrado em $projectRoot"
}

$rscriptPath = Get-RscriptPath

Write-Host "Usando Rscript em: $rscriptPath"
Write-Host "Executando: $mainScript"

Push-Location $projectRoot
try {
  & $rscriptPath --vanilla $mainScript @args
  if ($null -eq $LASTEXITCODE) {
    exit 0
  }

  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
