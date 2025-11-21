
# AoT-PsyPhyBENCH Metadata

This CSV file contains metadata and annotations for video clips used in our study, based on the dataset introduced by:

Hanyu et al. (2023), "Ready to detect a reversal of time’s arrow? A psychophysical study using short video clips in daily events" (2023)
https://royalsocietypublishing.org/doi/full/10.1098/rsos.230036
 

## 📂 File: `AoT-PsyPhyBENCH_dataset_metadata.csv`

- Total entries: 212 video clips
- Format: CSV (comma-separated values)

## 🧾 Column Structure

- 'id': Filename assigned by Hanyu et al. (2023).
- 'folder': Folder in which the video is located in the Multi-Moments in Time datasets.
- 'filename': Original filenames in the Multi-Moments in Time datasets.
- 'motion_category': Motion category annotation from Hanyu et al. (2023).

Please download the Multi-Moments in Time dataset, extract all files listed in the 'filename' column, and rename them to the values in 'id'. Save the renamed files in this directory.

## 📌 Note

Hanyu et al. (2023) used a version of the Moments in Time dataset from 2019, which appears to have since been updated.
We reused the same originally sampled videos. However, some videos are no longer available in the latest versions of the dataset.
We tried to match the videos with the Multi-Moments in Time dataset (an updated version of the Moments in Time dataset) in order to provide the original IDs of the files.

🚫 No video files are distributed in this repository. Only metadata and labels originally provided by the authors are included to support reproducibility and further research.

📥 To access the video files, please visit: [Multi-Moments in Time Dataset](http://moments.csail.mit.edu/)
For the no longer available videos, please contact us directly: 
matta@nlp.ist.i.kyoto-u.ac.jp or liskanashiro@nict.go.jp
