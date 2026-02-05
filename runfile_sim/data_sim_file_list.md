# data_sim file list

The copy scripts copy the **entire source Mat directory** into `data_sim`. Below are only the **data files actually used** by the five Python scripts (`data_folder` + filename; on disk the files are usually the same name with or without `.mat`).

---

## 1. Per .mat file: which .py uses it and which variable names

| .mat file | Used by .py | Variable names in code |
|-----------|-------------|------------------------|
| **dataset_split_index** | All 5 scripts | `spliting_index` (then `train_index`, `test_index`) |
| **Input_Bscan** | Para2outV10…, PiNet, MultipathEncoder, DMRF-UNet | `input_train` / `input_test` or `input_train` / `input_val` or `input1_train` / `input1_val` |
| **Output_LayerPerValue** | Para2outV10… | `output_train_layer`, `output_test_layer` |
| **Output_DefectPerValue** | Para2outV10… | `output_train_decay`, `output_test_decay` |
| **Output_DefectPerMap** | PiNet, MultipathEncoder, DMRF-UNet | `output_train`, `output_val` |
| **edge_info_v3000** | PiNet, MultipathEncoder, DMRF-UNet, Geo_UNetV11 | `crop_train`, `crop_val` |
| **Input_Migration** | MultipathEncoder (input2), Geo_UNetV11 (train input) | `input2_train`/`input2_val` or `input_train` |
| **Input_Migration_TestDataset** | Geo_UNetV11 | `input_val` (test set) |
| **Output_GeoMap** | Geo_UNetV11 | `output_train`, `output_val` |

---

## 2. Variable names read inside each .mat file

These are the **MATLAB variable names** (keys after `sio.loadmat`) that the Python code reads. `{index}` means a 5-digit sample index (e.g. 00001).

| .mat file | Variable names inside the .mat | Notes |
|-----------|-------------------------------|-------|
| **dataset_split_index** | `train_index`, `test_index` | Accessed as `spliting_index["train_index"]` / `spliting_index["test_index"]` |
| **Input_Bscan** | `rawwotz_twodefect_norm_{index}` | One variable per sample; index from train_index/test_index, zero-padded to 5 digits |
| **Output_LayerPerValue** | `er_ssim2_total_norm` | Single vector; subsets taken by train_index/test_index |
| **Output_DefectPerValue** | `per_defect_normV4_{index}` | One variable per sample, 5-digit index |
| **Output_DefectPerMap** | `defect_per_norm` + `{index}` (5 digits) | geo_prefix + 5-digit index, no suffix |
| **edge_info_v3000** | `edge_{index}` | One variable per sample, 5-digit index |
| **Input_Migration** (MultipathEncoder) | `entropy_twodefect_{index}v` | input2_prefix + 5 digits + input2_suffix |
| **Input_Migration** (Geo_UNetV11) | `ssim2_twodefect_{index}v` | migration_prefix + 5 digits + migration_suffix |
| **Input_Migration_TestDataset** | `ssim2_twodefect_{index}v` | Same pattern, indexed by test_index |
| **Output_GeoMap** | `defect_geo_{index}` | geo_prefix + 5 digits + geo_suffix (empty) |

**Index rule:** `{index}` is each integer in `train_index` or `test_index` formatted as a 5-digit string (e.g. `1` → `00001`).

---

## 3. By data file → which scripts need it

| Data file (in data_sim) | Purpose | Scripts |
|-------------------------|---------|---------|
| **dataset_split_index** | Train/test split indices | All 5 |
| **Input_Migration** | Migration images (train, ssim2) | Geo_UNetV11, MultipathEncoder |
| **Input_Migration_TestDataset** | Migration images (test, ssim2) | Geo_UNetV11 |
| **Output_GeoMap** | Geometry B-scan (labels) | Geo_UNetV11 |
| **edge_info_v3000** | Crop/edge info | Geo_UNetV11, DMRF-UNet, MultipathEncoder, PiNet |
| **Input_Bscan** | Normalized migration (rawwotz) | DMRF-UNet, MultipathEncoder, PiNet, Para2outV10 |
| **Output_DefectPerMap** | Defect permittivity labels | DMRF-UNet, MultipathEncoder, PiNet |
| **Output_DefectPerValue** | Defect decay labels | Para2outV10 |
| **Output_LayerPerValue** | Layer permittivity vector (er_ssim2_total_norm) | Para2outV10 |

---

## 4. By script → required data files

| .py script | Required data files |
|------------|----------------------|
| **MSE_2DV13_Geo_UNetV11_ssim2_v3000_Check_Xavier_ConstantLR.py** | dataset_split_index, Input_Migration, Input_Migration_TestDataset, Output_GeoMap, edge_info_v3000 |
| **MSE_2DV13_Per_DMRF-UNet_rawwotz_v3000_Check_Xavier.py** | dataset_split_index, Input_Bscan, Output_DefectPerMap, edge_info_v3000 |
| **MSE_2DV13_Per_MultipathEncoder_rawwotz_v3000_Check_Xavier.py** | dataset_split_index, Input_Bscan, Input_Migration, Output_DefectPerMap, edge_info_v3000 |
| **MSE_2DV13_Per_PiNet_rawwotz_v3000_Check_Xavier.py** | dataset_split_index, Input_Bscan, Output_DefectPerMap, edge_info_v3000 |
| **Para2outV10_rawwotz_norm_2output_2DV13_V3000.py** | dataset_split_index, Input_Bscan, Output_LayerPerValue, Output_DefectPerValue |

---

## 5. Full list (9 files)

1. dataset_split_index  
2. Input_Migration  
3. Input_Migration_TestDataset  
4. Output_GeoMap  
5. edge_info_v3000  
6. Input_Bscan  
7. Output_DefectPerMap  
8. Output_DefectPerValue  
9. Output_LayerPerValue  

Scripts use the names above; on disk the files may have a `.mat` extension (e.g. `dataset_split_index.mat`). The copy scripts copy the whole directory, so all of these can live under `data_sim`.
