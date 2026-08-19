# llm-stack.ps1 — supervisor for the local model servers (see LOCAL_SETUP.md)
#
#   .\llm-stack.ps1 start     start both servers under a supervisor (auto-restarts on crash/hang)
#   .\llm-stack.ps1 stop      kill switch: stops supervisor and both servers
#   .\llm-stack.ps1 restart   stop + start
#   .\llm-stack.ps1 status    processes, health, VRAM
#
# Servers:
#   llama.cpp :8090  smallest GGUF found in models\ (root reasoning model)
#   AirLLM    :8091  first safetensors checkpoint dir found in models\ (layer streaming, instrumented)
#   worker    :8092  smallest GGUF found in models\worker\ (fast leaf-task model for rlm sub-calls)
#
# Model discovery keeps this script model-agnostic: drop your checkpoints into
# models\ and the stack picks them up. Server flags (context size, KV cache type,
# flash attention, extra args) come from rlm-rs\rlm.config.json — the single
# source of truth shared with the Rust harness.

param([Parameter(Position = 0)][ValidateSet('start', 'stop', 'restart', 'status', 'supervise')][string]$cmd = 'status')

$root     = $PSScriptRoot
$rt       = "$root\runtime"
$stopFlag = "$rt\stack.stop"
$stackLog = "$rt\stack.log"

# Shared server configuration (falls back to the committed example).
$cfgPath = if (Test-Path "$root\rlm-rs\rlm.config.json") { "$root\rlm-rs\rlm.config.json" } else { "$root\rlm-rs\rlm.config.example.json" }
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json

# Smallest non-projector GGUF directly under models\ (quants sort below full precision),
# unless the config pins one.
$llamaModel = if ($cfg.model_path) { Get-Item (Join-Path $root $cfg.model_path) -ErrorAction SilentlyContinue } else {
    Get-ChildItem "$root\models\*.gguf" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^mmproj' } | Sort-Object Length | Select-Object -First 1
}
# AirLLM checkpoint: config pin wins; otherwise first subdirectory of models\
# holding a safetensors checkpoint.
$airllmModel = if ($cfg.airllm_model_path) { Get-Item (Join-Path $root $cfg.airllm_model_path) -ErrorAction SilentlyContinue } else {
    Get-ChildItem "$root\models" -Directory -ErrorAction SilentlyContinue |
        Where-Object { (Test-Path "$($_.FullName)\*.safetensors") -and ($_.Name -ne 'worker') } | Select-Object -First 1
}
# Worker model for rlm leaf sub-calls (optional).
$workerModel = if ($cfg.worker_model_path) { Get-Item (Join-Path $root $cfg.worker_model_path) -ErrorAction SilentlyContinue } else {
    Get-ChildItem "$root\models\worker\*.gguf" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^mmproj' } | Sort-Object Length | Select-Object -First 1
}

$llamaExe  = "$rt\llama.cpp\llama-server.exe"
$llamaArgs = "-m `"$($llamaModel.FullName)`" --host $($cfg.host) --port $($cfg.port) -c $($cfg.ctx_size) --jinja"
if ($cfg.flash_attn)    { $llamaArgs += " -fa on" }
if ($cfg.kv_cache_type) { $llamaArgs += " --cache-type-k $($cfg.kv_cache_type) --cache-type-v $($cfg.kv_cache_type)" }
if ($null -ne $cfg.n_gpu_layers) { $llamaArgs += " -ngl $($cfg.n_gpu_layers)" }
# Quote-wrap extra args containing specials (e.g. JSON values for --chat-template-kwargs)
# so they survive the cmd wrapper as single arguments.
if ($cfg.extra_server_args) {
    $quoted = $cfg.extra_server_args | ForEach-Object {
        if ($_ -match '[\s"{}]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    $llamaArgs += " " + ($quoted -join ' ')
}
$airllmPy   = "$root\.venv\Scripts\python.exe"
$airllmPort = 8091
$airllmArgs = "`"$root\serve_airllm.py`" --model `"$($airllmModel.FullName)`" --port $airllmPort"
$workerPort = if ($cfg.worker_port) { $cfg.worker_port } else { 8092 }
$workerArgs = "-m `"$($workerModel.FullName)`" --alias local-worker --chat-template-kwargs " + '"{\"enable_thinking\":false}"' + " --host $($cfg.host) --port $workerPort -c $($cfg.worker_ctx) --jinja"
if ($cfg.flash_attn)    { $workerArgs += " -fa on" }
if ($cfg.kv_cache_type) { $workerArgs += " --cache-type-k $($cfg.kv_cache_type) --cache-type-v $($cfg.kv_cache_type)" }
if ($null -ne $cfg.worker_n_gpu_layers) { $workerArgs += " -ngl $($cfg.worker_n_gpu_layers)" }

# Startup grace before health-based restarts kick in (model loading takes a while).
$graceSeconds = 600

function Log([string]$msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Encoding utf8 $stackLog
}

# Main and worker are both llama-server.exe — tell them apart by the port on their command line.
function Get-LlamaProcByPort([int]$port) {
    Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" |
        Where-Object { $_.CommandLine -match "--port $port\b" }
}
function Get-LlamaProc  { Get-LlamaProcByPort $cfg.port }
function Get-WorkerProc { Get-LlamaProcByPort $workerPort }
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
    $started = @{ llama = $null; airllm = $null; worker = $null }   # last (re)start time, for the health grace period
    $unhealthy = @{ llama = 0; airllm = 0; worker = 0 }

    $names = @('llama', 'airllm') + $(if ($workerModel) { @('worker') } else { @() })
    while (-not (Test-Path $stopFlag)) {
        foreach ($name in $names) {
            $proc = switch ($name) { 'llama' { Get-LlamaProc } 'airllm' { Get-AirllmProc } 'worker' { Get-WorkerProc } }
            $port = switch ($name) { 'llama' { $cfg.port } 'airllm' { $airllmPort } 'worker' { $workerPort } }

            if (-not $proc) {
                switch ($name) {
                    'llama'  { Start-Server "llama.cpp:$($cfg.port)" $llamaExe $llamaArgs "$rt\llama-server.log" }
                    'airllm' { Start-Server "airllm:$airllmPort"     $airllmPy  $airllmArgs "$rt\serve_airllm.log" }
                    'worker' { Start-Server "worker:$workerPort"     $llamaExe  $workerArgs "$rt\worker.log" }
                }
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
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
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
        Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Get-AirllmProc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Remove-Item $stopFlag -Force -ErrorAction SilentlyContinue
        Write-Output "stack stopped (supervisor + all servers)."
    }

    'restart' {
        & $PSCommandPath stop
        & $PSCommandPath start
    }

    'status' {
        $sup = Get-SupervisorProc
        Write-Output ("supervisor : " + $(if ($sup) { "running (pid $($sup.ProcessId))" } else { "not running" }))
        $l = Get-LlamaProc
        Write-Output ("llama:$($cfg.port) : " + $(if ($l) { "pid $($l.ProcessId), healthy=$(Test-Health $cfg.port)" } else { "down" }))
        $w = Get-WorkerProc
        Write-Output ("worker:$workerPort" + ": " + $(if ($w) { "pid $($w.ProcessId), healthy=$(Test-Health $workerPort)" } elseif ($workerModel) { "down" } else { "no worker model in models\worker\" }))
        $a = Get-AirllmProc
        Write-Output ("airllm:$airllmPort" + ": " + $(if ($a) { "pid $($a.ProcessId), healthy=$(Test-Health $airllmPort)" } else { "down" }))
        Write-Output ("VRAM       : " + (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader))
    }
}
