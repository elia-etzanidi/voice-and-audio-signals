# Speech Background Segmentation

This code implements the Voice Activity Detection task using k-NN and a 2-Layer MLP.

## Prerequisites

To run the code, install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Data Structure

The audio files and transcriptions are not included.

In order to run `main.py`, create a `data` folder in the root directory with the following structure:

```text
project_folder/
├── main.py
├── src/
├── data/
│   ├── train/
│   │   ├── noise/  (contains the free-sound & sound-bible subfolders)
│   │   └── speech/ (contains the librivox & us-gov subfolders)
│   └── test/
│       ├── S01_U04.CH4.wav
│       └── S01.json
└── requirements.txt
```

## Execution

Once the data has been placed in the appropriate directories, run:

```bash
python main.py
```

The program will read the data, train the models, and save the final CSV files in the `results/` folder, which will be created automatically.
