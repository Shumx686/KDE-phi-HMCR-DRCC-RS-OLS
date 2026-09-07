param(
    [Parameter(Mandatory = $true)][string]$NewCorrectionRoot,
    [Parameter(Mandatory = $true)][string]$BundleOutput
)
# Reproduce the correction in NEW directories; never rerun frozen optimizers.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$correctionRoot = [IO.Path]::GetFullPath($NewCorrectionRoot)
$bundleRoot = [IO.Path]::GetFullPath($BundleOutput)
if ((Test-Path -LiteralPath $correctionRoot) -or (Test-Path -LiteralPath $bundleRoot)) {
    throw 'Both output directories must be new; no original evidence is overwritten.'
}
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
Push-Location -LiteralPath $taskRoot
try {
    Invoke-Checked 'julia' @('--project=experiment_code', 'experiment_code/postprocess/test_test_wind_acceptance.jl')
    $chains = @(
        @('rbts_stress', 'rbts_stress_n1000', 'final_rbts_stress', 'RBTS', 'RBTS', '1001', '2000'),
        @('rts79_stress', 'rts79_stress_n500', 'final_rts79_stress', 'RTS79', 'RTS_79', '1', '500'),
        @('rbts_unconditional', 'rbts_unconditional_n500', 'final_rbts_unconditional', 'RBTS', 'RBTS', '1', '500'),
        @('rts79_unconditional', 'rts79_unconditional_n500', 'final_rts79_unconditional', 'RTS79', 'RTS_79', '1', '500')
    )
    $files = @{}
    foreach ($chain in $chains) {
        $name, $sourceDir, $stem, $system, $loadSystem, $first, $last = $chain
        $target = Join-Path $correctionRoot $name
        $inputPrefix = "experiment_code/results/$sourceDir/analysis/$stem"
        Invoke-Checked 'julia' @('--project=experiment_code', 'experiment_code/postprocess/apply_test_wind_acceptance_correction.jl',
            "${inputPrefix}_long.csv", "${inputPrefix}_manifest.toml", "experiment_code/$loadSystem/BUS_LOADS.csv", $target)
        $scope = if ($name.EndsWith('_stress')) { 'stress' } else { 'unconditional' }
        $analysis = Join-Path $target 'analysis'
        Invoke-Checked 'julia' @('--project=experiment_code', 'experiment_code/postprocess/analyze_final_frozen_replays.jl',
            $system, $scope, (Join-Path $target 'corrected_inputs'), $first, $last, $analysis)
        $prefix = Join-Path $analysis $stem
        if ($scope -eq 'unconditional') {
            $hybrid = Join-Path $target 'hybrid_analysis'
            Invoke-Checked 'julia' @('--project=experiment_code', 'experiment_code/postprocess/build_hybrid_reliability_summary.jl',
                $system, "${prefix}_long.csv", "${prefix}_manifest.toml", $hybrid,
                "experiment_code/results/$sourceDir/hybrid_analysis/${stem}_long.csv")
            $prefix = Join-Path $hybrid $stem
        }
        $files[$name] = $prefix
    }
    $rbtsStress = $files['rbts_stress']; $rtsStress = $files['rts79_stress']
    $rbtsAdeq = $files['rbts_unconditional']; $rtsAdeq = $files['rts79_unconditional']
    Invoke-Checked 'julia' @('--project=experiment_code', 'experiment_code/postprocess/generate_final_result_tables.jl',
        "${rbtsStress}_summary.csv", "${rbtsAdeq}_summary.csv", "${rtsStress}_summary.csv", "${rtsAdeq}_summary.csv", $bundleRoot)
    Invoke-Checked 'python' @('experiment_code/postprocess/generate_final_pairwise_claims.py',
        '--rbts-long', "${rbtsStress}_long.csv", '--rts79-long', "${rtsStress}_long.csv",
        '--output', (Join-Path $bundleRoot 'final_pairwise_claims.csv'),
        '--directional-output', (Join-Path $bundleRoot 'final_directional_event_rates.csv'))
    Invoke-Checked 'python' @('experiment_code/postprocess/generate_final_adequacy_claims.py',
        '--rbts-long', "${rbtsAdeq}_long.csv", '--rts79-long', "${rtsAdeq}_long.csv",
        '--output', (Join-Path $bundleRoot 'final_adequacy_pairwise_claims.csv'))
    Invoke-Checked 'python' @('experiment_code/postprocess/generate_final_result_figures.py',
        '--rbts-stress-summary', "${rbtsStress}_summary.csv", '--rbts-stress-long', "${rbtsStress}_long.csv",
        '--rts79-stress-summary', "${rtsStress}_summary.csv", '--rts79-stress-long', "${rtsStress}_long.csv",
        '--rbts-unconditional-summary', "${rbtsAdeq}_summary.csv", '--rts79-unconditional-summary', "${rtsAdeq}_summary.csv",
        '--output-dir', $bundleRoot)
} finally {
    Pop-Location
}
