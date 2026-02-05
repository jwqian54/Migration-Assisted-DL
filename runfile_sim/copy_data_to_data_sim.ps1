# Copy the 9 data files required by runfile_sim scripts into data_sim
# If you get an execution policy error, use copy_data_to_data_sim.bat instead

$Source = "D:\NTU_staff_OneDrive\OneDrive - Nanyang Technological University\Dataset\migration_image_ArbDefect_3D\output_mask\result\Mat"
$Dest   = "d:\tgrs_open\sim\data_sim"

$FileList = @(
    "dataset_split_index",
    "Input_Migration",
    "Input_Migration_TestDataset",
    "Output_GeoMap",
    "edge_info_v3000",
    "Input_Bscan",
    "Output_DefectPerMap",
    "Output_DefectPerValue",
    "Output_LayerPerValue"
)

if (-not (Test-Path $Source)) {
    Write-Error "Source path not found; ensure OneDrive is synced: $Source"
    exit 1
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "Copying 9 required data files to data_sim..."

foreach ($base in $FileList) {
    $withMat = Join-Path $Source "$base.mat"
    $noExt   = Join-Path $Source $base
    $destPath = Join-Path $Dest $base
    if (Test-Path $withMat) {
        Copy-Item -Path $withMat -Destination $destPath -Force
        Write-Host "  OK $base"
    } elseif (Test-Path $noExt) {
        Copy-Item -Path $noExt -Destination $destPath -Force
        Write-Host "  OK $base"
    } else {
        Write-Host "  Skipped (not found) $base"
    }
}

Write-Host "Done. The five .py scripts are set to read from data_sim."
