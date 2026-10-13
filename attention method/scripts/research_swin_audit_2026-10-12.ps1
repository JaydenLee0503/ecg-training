# Read saved artifacts only; writes one separate research JSON, never trains.
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$attention = Join-Path $repo 'attention method'
$reportDir = Join-Path $attention 'reports/ecgdata_lfcc_swin_v1'
$runDir = Join-Path $attention 'results/ecgdata_lfcc_swin_v1'
$inventoryPath = Join-Path $reportDir 'artifact_inventory.csv'
$inventory = @(Import-Csv -LiteralPath $inventoryPath)
$problems = @()
foreach ($entry in $inventory) {
    $path = Join-Path $runDir $entry.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $problems += [ordered]@{ path = $entry.path; issue = 'missing' }
        continue
    }
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne [long]$entry.bytes -or $hash -ne $entry.sha256) {
        $problems += [ordered]@{ path = $entry.path; issue = 'size_or_hash_mismatch' }
    }
}
$completion = Get-Content -LiteralPath (Join-Path $runDir 'completed.json') -Raw | ConvertFrom-Json
$saved = Get-Content -LiteralPath (Join-Path $reportDir 'report.json') -Raw | ConvertFrom-Json
$predictions = @(Import-Csv -LiteralPath (Join-Path $reportDir 'predictions.csv') | Where-Object arm -eq 'swin')
$fits = @(Import-Csv -LiteralPath (Join-Path $reportDir 'fits.csv') | Where-Object trial -like 'swin_*')
$histories = @(Import-Csv -LiteralPath (Join-Path $reportDir 'histories.csv'))
$classes = @('ARR', 'CHF', 'NSR')
$classStats = @()
for ($i = 0; $i -lt $classes.Count; $i++) {
    $classStats += [ordered]@{
        class = $classes[$i]
        mean_window_recall = ($saved.results.swin.per_seed | ForEach-Object { $_.window.sensitivity[$i] } | Measure-Object -Average).Average
        mean_window_f1 = ($saved.results.swin.per_seed | ForEach-Object { $_.window.f1[$i] } | Measure-Object -Average).Average
        mean_patient_recall = ($saved.results.swin.per_seed | ForEach-Object { $_.patient_vote.sensitivity[$i] } | Measure-Object -Average).Average
    }
}
$seedChecks = @()
$patientVotes = @()
foreach ($seed in 0, 1, 2) {
    $rows = @($predictions | Where-Object { [int]$_.seed -eq $seed })
    $patients = @($rows | Group-Object patient)
    $accuracy = @($rows | Where-Object { $_.label -eq $_.prediction }).Count / $rows.Count
    if ($rows.Count -ne 1620 -or @($rows.window_id | Sort-Object -Unique).Count -ne 1620 -or $patients.Count -ne 80) {
        throw 'Unexpected saved prediction coverage'
    }
    if ([math]::Abs($accuracy - $saved.results.swin.per_seed[$seed].window.accuracy) -gt 1e-12) {
        throw 'Saved window accuracy does not match prediction export'
    }
    $seedChecks += [ordered]@{
        seed = $seed; rows = $rows.Count
        unique_windows = @($rows.window_id | Sort-Object -Unique).Count
        patients = $patients.Count
        accuracy_recomputed_from_saved_classes = $accuracy
    }
    foreach ($patient in $patients) {
        $labels = @($patient.Group.label | Sort-Object -Unique)
        $folds = @($patient.Group.fold | Sort-Object -Unique)
        if ($labels.Count -ne 1 -or $folds.Count -ne 1) { throw 'Patient label/fold inconsistency' }
        $counts = @($classes | ForEach-Object { $class = $_; @($patient.Group | Where-Object prediction -eq $class).Count })
        $maxCount = ($counts | Measure-Object -Maximum).Maximum
        $vote = $classes[[array]::IndexOf($counts, [int]$maxCount)]
        $patientVotes += [ordered]@{
            seed = $seed; patient = $patient.Name; label = $labels[0]; vote = $vote
            correct = $vote -eq $labels[0]; windows = $patient.Count
        }
    }
    $seedVotes = @($patientVotes | Where-Object { $_.seed -eq $seed })
    $voteAccuracy = @($seedVotes | Where-Object { $_.correct }).Count / $seedVotes.Count
    if ([math]::Abs($voteAccuracy - $saved.results.swin.per_seed[$seed].patient_vote.accuracy) -gt 1e-12) {
        throw 'Recomputed patient voting disagrees with saved report'
    }
}
$sourcePaths = @(
    'attention method/ecgdata_protocol.json',
    'attention method/attention_ecg/model.py',
    'attention method/attention_ecg/cepstral.py',
    'attention method/attention_ecg/ecgdata_training.py',
    'attention method/reports/ecgdata_lfcc_swin_v1/report.json',
    'attention method/reports/ecgdata_lfcc_swin_v1/predictions.csv',
    'attention method/reports/ecgdata_lfcc_swin_v1/fits.csv',
    'attention method/reports/ecgdata_lfcc_swin_v1/histories.csv',
    'attention method/reports/ecgdata_lfcc_swin_v1/artifact_inventory.csv'
)
$sources = @($sourcePaths | ForEach-Object {
    $path = Join-Path $repo $_
    [ordered]@{ path = $_; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$result = [ordered]@{
    review_date_toronto = '2026-10-12'
    audited_at_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'Saved Swin v1 artifact checksum verification and descriptive analysis; no model loaded, fitted, or newly inferred'
    completion_status = $completion.status
    artifact_inventory_entries = $inventory.Count
    artifact_problems = @($problems)
    swin_fits = $fits.Count
    fits_with_perfect_training_accuracy = @($fits | Where-Object { [double]$_.training_accuracy -eq 1.0 }).Count
    history_rows = $histories.Count
    class_statistics_from_saved_report = $classStats
    prediction_coverage = $seedChecks
    incorrect_patient_votes = @($patientVotes | Where-Object { -not $_.correct })
    input_hashes = $sources
}
$output = Join-Path $attention 'reports/swin_research_audit_2026-10-12.json'
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $output -Encoding utf8
$result | ConvertTo-Json -Depth 12
if ($completion.status -ne 'complete' -or $problems.Count -gt 0) { throw 'Artifact verification failed; preserved all inputs' }
