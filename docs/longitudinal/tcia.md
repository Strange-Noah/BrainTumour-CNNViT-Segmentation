# UCSD-BMETS metadata audit and acquisition

## Published evidence

TCIA Version 1 dates to 2026-10-05. The collection describes 300 patients with two to four scans each and 1,065 MRI timepoints. Its download table reports 1,066 studies and 11,405 images, approximately 163 GB. Reconcile the counts from source inventories before declaring completion. The page lists four TSVs (clinical, MRI and their dictionaries). [TCIA collection](https://www.cancerimagingarchive.net/collection/ucsd-bmets-longitudinal/)

Examination IDs use `BMETSXXXX_YY`. TCIA describes precontrast T1, contrast-enhanced T1, T2 and FLAIR as core modalities with sequence tokens `t1n`, `t1c`, `t2` and `t2f`. Additional modalities vary. TCIA describes native-space images registered to first-visit contrast-enhanced T1 and calls its `-seg.nii.gz` masks binary. The first downloaded mask contained labels 0, 2 and 3, so the release cannot be treated as strictly binary. Class meanings need confirmation from TCIA. Preserve raw labels and define any research conversion as a versioned derivative.

## Official metadata audit

The four direct links were discovered from the official collection in a private Kaggle CPU run. Each direct URL timed out from the local runtime. Kaggle retrieved all four originals. A second independent request confirmed the clinical dictionary as a two-column TSV with a different header layout.

| Official file | Result |
|---|---|
| [Clinical data](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data.tsv) | Windows-1252, 300 unique case IDs and 170 blank spreadsheet rows. SHA-256 `d701179cba86124d833de30bed496337c19f054c49e48d6ebc25137542daf0a2`. |
| [Clinical data dictionary](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data_dictionary.tsv) | Windows-1252, 30 two-column rows including its header. SHA-256 `77fd461369f8b0e1f9bc762e79d6346d0a0f733cc5e4923ddbe8de17b65cb3a1`. |
| [MRI metadata](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_MRImetadata.tsv) | UTF-8, 1,065 unique study IDs, 300 patients, and zero unmatched patients in either direction. SHA-256 `53ac850fcd31094b2fc131cad08b23497b9539bbf75cf2b3679b7e7108fa41c7`. |
| [MRI metadata dictionary](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_MRImetadata_Dictionary.tsv) | Windows-1252, 12 entries. SHA-256 `b2647b0f8936676208bb3e7bc986eb8888f76376ea83d29412da82074ec51797`. |

The MRI TSV contains 49 patients with two visits, 37 with three, and 214 with four. Every study marks `t1n`, `t1c`, `t2w`, and `t2f` as present. The browsed patient files use the T2 filename token `t2w`, while the collection prose lists `t2`. The manifest keeps `t2` as the modality key and uses `t2w` in the expected filename. The MRI TSV reconciles the 1,065 timepoints described in the article, while the download table lists 1,066 studies.

The original clinical table contains pseudonymous case IDs, demographics, diagnosis, ages at imaging or death and treatment flags. The MRI table contains pseudonymous study IDs, scanner metadata and sequence availability. A scan found no email-like values or ISO-like calendar dates in either table. TCIA describes anonymized subject identifiers and skull-stripped scans. The original data remain participant-level research data and must be handled under the TCIA terms, including attribution and prohibitions on re-identification. [TCIA usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)

The minimized manifest joins `Study ID` to `Case ID` and uses ordinal timing. It excludes demographics, diagnoses, ages and treatment values. The 1,065 records retain pseudonymous linkage, expected paths and integrity status. All four original TSVs and the minimized manifest were stored in `novo-nordisk-pbl` and verified by SHA-256 readback. Manifest key `novo-nordisk/datasets/ucsd-bmets/v1/manifests/040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4/visits.jsonl`. Originals are under `novo-nordisk/datasets/ucsd-bmets/v1/metadata/`.

## Complete patient pilot

Private Kaggle CPU run 5 transferred all 43 files for one four-visit patient to ephemeral Linux staging using the official Faspex link. All files matched the source inventory sizes, totaling 649,354,820 bytes. The SHA-256 of the ordered filename and locally computed checksum pairs was `2d3fb187783099d98269bdece35d73eab9d970588d11346bc16bb197598446da`. Publisher checksums were unavailable to this run.

| Visit | Label 0 | Label 2 | Label 3 | Grid matches visit T1c |
|---|---:|---:|---:|---|
| 1 | 48,233,577 | 862 | 57 | Yes |
| 2 | 48,233,545 | 859 | 92 | Yes |
| 3 | 48,233,700 | 717 | 79 | Yes |
| 4 | 48,233,802 | 662 | 32 | Yes |

The pilot published aggregate evidence. Patient imaging was held in temporary Kaggle staging and still needs a persistent R2 upload. Use a separate cloud ingestion runtime with an authorized dataset writer so the Kaggle dataset credentials remain read-only. Start by repeating the 43-file pilot into persistent staging and verifying R2 readback. Obtain sufficient staging capacity and review the inventory and write path before ingesting the full collection. [Private pilot, version 5](https://www.kaggle.com/code/mathewzhou/novo-tcia-package-inventory)

## Acquisition procedure to validate

1. Rebuild the pending manifest with the verified MRI and clinical originals.

   ```sh
   python cloud/manifest.py MRI.tsv CLINICAL.tsv visits.jsonl \
     --exam-column 'Study ID' --patient-column 'Case ID' --clinical-encoding cp1252
   ```

2. Use the official imaging row, which links to `faspex.cancerimagingarchive.net`. IBM supports receiving Faspex external package links with the Aspera CLI and a transfer agent. IBM Aspera CLI v4.28.0 successfully browsed the official Faspex public link from Kaggle CPU and found the root directory `/UCSD-BMETS-Longitudinal`. Browsing the collection directory returned 1,065 examination folders, an exact ID match to the MRI TSV. One patient had four examination folders with 43 files totaling 649,354,820 bytes, including four masks and all four core modalities at each visit. Install the IBM Transfer SDK with `ascli config ascp install`. Then use `ascli faspex5 packages receive` with the official public URL, an explicit `--sources=@json:...` array of inventoried paths, and `--to-folder` pointed at private Linux staging. A selective transfer retrieved a mask and its T1c MRI at the official sizes. Both NIfTI grids matched. The mask had 48,233,577 voxels of label 0, 862 of label 2 and 57 of label 3. Confirm the semantics and availability of publisher-provided checksums with TCIA. [TCIA support](https://www.cancerimagingarchive.net/support/) · [IBM Aspera CLI](https://github.com/IBM/aspera-cli/releases/tag/v4.28.0) · [Private Kaggle inventory probe](https://www.kaggle.com/code/mathewzhou/novo-tcia-package-inventory)
3. Pick a patient whose complete visit set is documented in the verified metadata. Obtain all original files for that patient in the authorized Linux runtime, with persisted staging for interrupted transfers. Confirm source counts, expected sizes and available published hashes. Run `python cloud/audit_visit.py MRI.nii.gz MASK.nii.gz` for each visit. Inspect observed values, affine, axis codes and units before choosing any label mapping.
4. Run `python cloud/ingest.py STAGE OFFICIAL_SIZE_INVENTORY.json RECEIPT.jsonl`. The inventory maps each relative path to its expected byte count from the authorized acquisition records. Use `--area metadata` for the TSV files. Reuse staged files and use a new receipt path for each attempt. Existing raw objects must match the SHA-256 on readback. Full ingest proceeds after the patient sequence passes, the full inventory and capacity are confirmed, and the cloud transfer path is proven. Preserve original archive members and ordering in provenance.
5. Write the reviewed manifest revision after verifying actual object keys and geometry. Record immutable manifest and acquisition receipts in R2 and commit their digests and aggregate counts in GitHub.

Correia de Verdier, M., Saluja, R., Gupta, D., Hattangadi-Gluth, J., Beaumont, T., Seibert, T., Dale, A., Farid, N., Gagnon, L., and Rudie, J. (2026). *The University of California San Diego Longitudinal Brain Metastases Annotated Multimodal MRI Dataset*, version 1. The Cancer Imaging Archive. CC BY 4.0. Follow TCIA Data Usage Policy and retain attribution in derivatives. [Dataset DOI](https://doi.org/10.7937/5ejb-9g94)
