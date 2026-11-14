# UCSD-BMETS metadata audit and acquisition

## Published evidence

TCIA Version 1 dates to 2026-10-05. The collection describes 300 patients with two to four scans each and 1,065 MRI timepoints. Its download table reports 1,066 studies and 11,405 images, approximately 163 GB. Reconcile the counts from source inventories before declaring completion. The page lists four TSVs (clinical, MRI and their dictionaries). [TCIA collection](https://www.cancerimagingarchive.net/collection/ucsd-bmets-longitudinal/)

Examination IDs use `BMETSXXXX_YY`. TCIA describes precontrast T1, contrast-enhanced T1, T2 and FLAIR as core modalities with sequence tokens `t1n`, `t1c`, `t2` and `t2f`. Additional modalities vary. TCIA describes native-space images registered to first-visit contrast-enhanced T1 and binary masks with `-seg.nii.gz` suffix. Actual label integers, MRI TSV columns, relative timing precision, treatment fields and NIfTI grid compatibility require file-level verification.

## Official metadata audit

The four direct links were discovered from the official collection in a private Kaggle CPU run. Each direct URL timed out from the local runtime. Kaggle retrieved and validated three originals. The clinical dictionary response failed TSV validation and needs follow-up.

| Official file | Result |
|---|---|
| [Clinical data](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data.tsv) | Windows-1252, 300 unique case IDs and 170 blank spreadsheet rows. SHA-256 `d701179cba86124d833de30bed496337c19f054c49e48d6ebc25137542daf0a2`. |
| [Clinical data dictionary](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_Clinical_Data_dictionary.tsv) | Kaggle received bytes. The response did not parse as a tabular TSV. |
| [MRI metadata](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_TCIA_MRImetadata.tsv) | UTF-8, 1,065 unique study IDs, 300 patients, and zero unmatched patients in either direction. SHA-256 `53ac850fcd31094b2fc131cad08b23497b9539bbf75cf2b3679b7e7108fa41c7`. |
| [MRI metadata dictionary](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD-BMETS-Longitudinal_MRImetadata_Dictionary.tsv) | Windows-1252, 12 entries. SHA-256 `b2647b0f8936676208bb3e7bc986eb8888f76376ea83d29412da82074ec51797`. |

The MRI TSV contains 49 patients with two visits, 37 with three, and 214 with four. Every study marks `t1n`, `t1c`, `t2w`, and `t2f` as present. The collection names the corresponding T2 file token `t2`. File paths remain expected until checked against the package listing. The MRI TSV reconciles the 1,065 timepoints described in the article, while the download table lists 1,066 studies.

The manifest joins `Study ID` to `Case ID` and uses ordinal timing. Clinical imaging ages and radiation intervals reference clinical events, so they are not converted into per-exam day offsets without a verified mapping. The pending manifest has SHA-256 `2c69309f3f122e8c6ab2f031450770325fb5eeea4776ffa2e9ec2cd79cd0d76b`. Automatic approval review blocked upload of the clinical metadata and derived manifest to R2 because the payloads may contain sensitive health information.

## Acquisition procedure to validate

1. Rebuild the pending manifest with the verified MRI and clinical originals.

   ```sh
   python cloud/manifest.py MRI.tsv CLINICAL.tsv visits.jsonl \
     --exam-column 'Study ID' --patient-column 'Case ID' --clinical-encoding cp1252 \
     --treatment-column Resection \
     --treatment-column 'Radiation before first MRI' \
     --treatment-column 'Radiation treatment between first and second MRI' \
     --treatment-column 'Radiation treatment between second and third MRI' \
     --treatment-column 'Radiation treatment between third and fourth MRI'
   ```

2. Use the official imaging row, which links to `faspex.cancerimagingarchive.net`. IBM supports receiving Faspex external package links with the Aspera CLI and a transfer agent. A direct HTTP file endpoint, collection-specific headless package receipt, and patient selection are unverified. Ask TCIA support for the supported Linux download procedure, package inventory, checksums and a way to select one complete patient before starting a bounded transfer. [TCIA support](https://www.cancerimagingarchive.net/support/) · [IBM Aspera CLI](https://www.ibm.com/support/pages/ibm-aspera-cli-v4x-available)
3. Pick a patient whose complete visit set is documented in the verified metadata. Obtain all original files for that patient in the authorized Linux runtime, with persisted staging for interrupted transfers. Confirm source counts, expected sizes and available published hashes. Run `python cloud/audit_visit.py MRI.nii.gz MASK.nii.gz` for each visit. Inspect observed values, affine, axis codes and units before choosing any label mapping.
4. Run `python cloud/ingest.py STAGE OFFICIAL_SIZE_INVENTORY.json RECEIPT.jsonl`. The inventory maps each relative path to its expected byte count from the authorized acquisition records. Use `--area metadata` for the TSV files. Reuse staged files and use a new receipt path for each attempt. Existing raw objects must match the SHA-256 on readback. Full ingest proceeds after the patient sequence passes, the full inventory and capacity are confirmed, and the cloud transfer path is proven. Preserve original archive members and ordering in provenance.
5. Write the reviewed manifest revision after verifying actual object keys and geometry. Record immutable manifest and acquisition receipts in R2 and commit their digests and aggregate counts in GitHub.

Correia de Verdier, M., Saluja, R., Gupta, D., Hattangadi-Gluth, J., Beaumont, T., Seibert, T., Dale, A., Farid, N., Gagnon, L., and Rudie, J. (2026). *The University of California San Diego Longitudinal Brain Metastases Annotated Multimodal MRI Dataset*, version 1. The Cancer Imaging Archive. CC BY 4.0. Follow TCIA Data Usage Policy and retain attribution in derivatives. [Dataset DOI](https://doi.org/10.7937/5ejb-9g94)
