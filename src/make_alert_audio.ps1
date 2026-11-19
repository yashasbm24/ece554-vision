param([string]$OutputDirectory = "models/alerts")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$streamType = [Windows.Media.SpeechSynthesis.SpeechSynthesisStream,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
$directory = (New-Item -ItemType Directory -Force $OutputDirectory).FullName
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
try {
    foreach ($label in @("person", "bicycle", "car", "traffic light", "stop sign", "bench", "dog", "backpack", "chair")) {
        $task = $asTask.MakeGenericMethod($streamType).Invoke($null, @($synth.SynthesizeTextToStreamAsync($label)))
        $task.Wait()
        $stream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($task.Result)
        $file = [IO.File]::Create((Join-Path $directory ($label.Replace(" ", "_") + ".wav")))
        try { $stream.CopyTo($file) } finally { $file.Dispose(); $stream.Dispose() }
    }
} finally { $synth.Dispose() }
