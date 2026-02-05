@echo off
chcp 65001 >nul
REM Copy the 9 data files required by runfile_sim scripts into data_sim
REM Run by double-click or in cmd: runfile_sim\copy_data_to_data_sim.bat

set "Source=D:\NTU_staff_OneDrive\OneDrive - Nanyang Technological University\Dataset\migration_image_ArbDefect_3D\output_mask\result\Mat"
set "Dest=d:\tgrs_open\sim\data_sim"

if not exist "%Source%" (
    echo Source path not found; ensure OneDrive is synced:
    echo %Source%
    pause
    exit /b 1
)

if not exist "%Dest%" mkdir "%Dest%"
echo Copying 9 required data files to data_sim...

set "LIST=dataset_split_index Input_Migration Input_Migration_TestDataset Output_GeoMap edge_info_v3000 Input_Bscan Output_DefectPerMap Output_DefectPerValue Output_LayerPerValue"

for %%f in (%LIST%) do (
    if exist "%Source%\%%f.mat" (
        copy /Y "%Source%\%%f.mat" "%Dest%\%%f" >nul && echo   OK %%f
    ) else if exist "%Source%\%%f" (
        copy /Y "%Source%\%%f" "%Dest%\%%f" >nul && echo   OK %%f
    ) else (
        echo  Skipped (not found) %%f
    )
)

echo Done. The five .py scripts are set to read from data_sim.
pause
