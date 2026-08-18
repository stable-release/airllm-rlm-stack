# llm-stack.ps1 — supervisor for the local model servers (see LOCAL_SETUP.md)
#
#   .\llm-stack.ps1 start     start both servers under a supervisor (auto-restarts on crash/hang)
#   .\llm-stack.ps1 stop      kill switch: stops supervisor and both servers
#   .\llm-stack.ps1 restart   stop + start
#   .\llm-stack.ps1 status    processes, health, VRAM
#
# Servers:
#   llama.cpp :8090  smallest GGUF found in models\ (interactive speed)
#   AirLLM    :8091  first safetensors checkpoint dir found in models\ (layer streaming, instrumented)
#
# Model discovery keeps this script model-agnostic: drop your checkpoints into
# models\ and the stack picks them up. Pin specific files by editing the two
# discovery lines below.

param([Parameter(Position = 0)][ValidateSet('start', 'stop', 'restart', 'status', 'supervise')][string]$cmd = 'status')

$root     = $PSScriptRoot
$rt       = "$root\runtime"
$stopFlag = "$rt\stack.stop"
$stackLog = "$rt\stack.log"

# Smallest non-projector GGUF directly under models\ (quants sort below full precision).
$llamaModel = Get-ChildItem "$root\models\*.gguf" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch '^mmproj' } | Sort-Object Length | Select-Object -First 1
# First subdirectory of models\ that holds a safetensors checkpoint.
$airllmModel = Get-ChildItem "$root\models" -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path "$($_.FullName)\*.safetensors" } | Select-Object -First 1

$llamaExe   = "$rt\llama.cpp\llama-server.exe"
$llamaArgs  = "-m `"$($llamaModel.FullName)`" --host 127.0.0.1 --port 8090 -c 32768 --jinja -fa on --cache-type-k q8_0 --cache-type-v q8_0"
$airllmPy   = "$root\.venv\Scripts\python.exe"
$airllmArgs = "`"$root\serve_airllm.py`" --model `"$($airllmModel.FullName)`" --port 8091"

# Startup grace before health-based restarts kick in (model loading takes a while).
$graceSeconds = 600

function Log([string]$msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Encoding utf8 $stackLog
}

function Get-LlamaProc  { Get-Process llama-server -ErrorAction SilentlyContinue }
function Get-AirllmProc {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'serve_airllm\.py' }
}
function Get-SupervisorProc {
    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -match 'llm-stack\.ps1.+supervise' -and $_.ProcessId -ne $PID }
}

function Test-Health([int]$port) {
    try { $null = Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3; $true }
    catch { $false }
}

function Start-Server([string]$name, [string]$exe, [string]$argLine, [string]$logFile) {
    Log "starting $name"
    # cmd wrapper gives us merged, appending logs across restarts.
    Start-Process cmd -ArgumentList '/c', "`"`"$exe`" $argLine >> `"$logFile`" 2>&1`"" -WindowStyle Hidden
}

function Supervise {
    Log "supervisor up (pid $PID)"
    $started = @{ llama = $null; airllm = $null }   # last (re)start time, for the health grace period
    $unhealthy = @{ llama = 0; airllm = 0 }

    while (-not (Test-Path $stopFlag)) {
        foreach ($name in @('llama', 'airllm')) {
            $proc = if ($name -eq 'llama') { Get-LlamaProc } else { Get-AirllmProc }
            $port = if ($name -eq 'llama') { 8090 } else { 8091 }

            if (-not $proc) {
                if ($name -eq 'llama') { Start-Server 'llama.cpp:8090' $llamaExe $llamaArgs "$rt\llama-server.log" }
                else                   { Start-Server 'airllm:8091'   $airllmPy  $airllmArgs "$rt\serve_airllm.log" }
                $started[$name] = Get-Date
                $unhealthy[$name] = 0
                continue
            }

            # Health-based restart: alive but unresponsive well past startup grace -> kill, loop restarts it.
            if ($started[$name] -and ((Get-Date) - $started[$name]).TotalSeconds -gt $graceSeconds) {
                if (Test-Health $port) {
                    $unhealthy[$name] = 0
                } else {
                    $unhealthy[$name]++
                    if ($unhealthy[$name] -ge 4) {
                        Log "$name unhealthy $($unhealthy[$name])x, killing for restart"
                        $proc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                        $unhealthy[$name] = 0
                    }
                }
            }
        }
        Start-Sleep 15
    }

    Log "stop flag seen; shutting servers down"
    Get-LlamaProc | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-AirllmProc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Log "supervisor exiting"
}

switch ($cmd) {
    'supervise' { Supervise }

    'start' {
        if (Get-SupervisorProc) { Write-Output "already running — use status"; break }
        if (-not $llamaModel)  { Write-Output "no .gguf found in models\ — add a model first"; break }
        Remove-Item $stopFlag -Force -ErrorAction SilentlyContinue
        Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$root\llm-stack.ps1", 'supervise' -WindowStyle Hidden
        Write-Output "stack starting. llama.cpp:8090 ($($llamaModel.Name)); airllm:8091 ($(if($airllmModel){$airllmModel.Name}else{'no safetensors dir found'})). Check: .\llm-stack.ps1 status"
    }

    'stop' {
        New-Item -ItemType File -Force $stopFlag | Out-Null
        # Give the supervisor one poll cycle to shut things down in order...
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-SupervisorProc) -and (Get-Date) -lt $deadline) { Start-Sleep 2 }
        # ...then guarantee the result regardless.
        Get-SupervisorProc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Get-LlamaProc | Stop-Process -Force -ErrorAction SilentlyContinue
        Get-AirllmProc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Remove-Item $stopFlag -Force -ErrorAction SilentlyContinue
        Write-Output "stack stopped (supervisor + both servers)."
    }

    'restart' {
        & $PSCommandPath stop
        & $PSCommandPath start
    }

    'status' {
        $sup = Get-SupervisorProc
        Write-Output ("supervisor : " + $(if ($sup) { "running (pid $($sup.ProcessId))" } else { "not running" }))
        $l = Get-LlamaProc
        Write-Output ("llama:8090 : " + $(if ($l) { "pid $($l.Id), healthy=$(Test-Health 8090)" } else { "down" }))
        $a = Get-AirllmProc
        Write-Output ("airllm:8091: " + $(if ($a) { "pid $($a.ProcessId), healthy=$(Test-Health 8091)" } else { "down" }))
        Write-Output ("VRAM       : " + (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader))
    }
}
