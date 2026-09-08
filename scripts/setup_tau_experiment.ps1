param(
    [string]$Commit = "672227c6b6676edc20d57ea53b7000262aae77b9"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Checkout = Join-Path $ProjectRoot "data/raw/tau/tau2-bench"
$Uv = (Get-Command uv -ErrorAction Stop).Source

if (-not (Test-Path (Join-Path $Checkout ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $Checkout) | Out-Null
    git clone https://github.com/sierra-research/tau2-bench $Checkout
}

$Actual = (git -C $Checkout rev-parse HEAD).Trim()
if ($Actual -ne $Commit) {
    git -C $Checkout fetch origin $Commit
    git -C $Checkout checkout --detach $Commit
}

& $Uv venv --python 3.12 (Join-Path $ProjectRoot ".venv-tau")
$Python = Join-Path $ProjectRoot ".venv-tau/Scripts/python.exe"
& $Uv pip install --python $Python -r (Join-Path $ProjectRoot "requirements-tau.txt")
& $Python -c "import tau2; print('tau2 import: OK')"
