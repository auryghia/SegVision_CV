# SegVision_CV# SegVision_CV: Fine-tuning SAM on COCO 2017

This project demonstrates how to fine-tune the Segment Anything Model (SAM) on a custom subset of the COCO 2017 dataset. The primary focus is on segmenting a specific class (e.g., "person") by fine-tuning only the mask decoder of SAM, leveraging precomputed image embeddings to speed up the training process.

## Project Structure

```
SegVision_CV/
├── Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb  # Main Jupyter Notebook
├── precomputed_embeddings.pt       # Example of precomputed embeddings file
├── final_model.pth                 # Example of a saved fine-tuned model
├── training_SAM_results.txt        # Example of training log
├── utils/                            # Utility scripts
│   ├── dataloader.py               # COCO dataset and dataloader logic
│   ├── helpers.py                  # EmbeddingDataset class
│   ├── metrics.py                  # Dice metric calculation
│   └── visualization.py            # Visualization functions
├── models/                           # Model definitions
│   └── SAM.py                      # SAM model wrapper
├── data/                             # (Gitignored) COCO dataset storage
│   └── train2017/
│       ├── annotations/
│       └── images/
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Features

- Fine-tuning of SAM's mask decoder.
- Usage of precomputed image embeddings for efficient training.
- Custom data loading for COCO 2017, focusing on specific classes.
- Implementation of DiceCELoss for training and Dice coefficient for evaluation.
- Visualization of images, ground truth masks, and predicted masks.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd SegVision_CV
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Note: If you face issues with the `transformers` library, you might need to install it directly from the Hugging Face GitHub repository as done in the notebook:_

    ```bash
    pip install git+https://github.com/huggingface/transformers.git
    ```

4.  **Download COCO 2017 Dataset:**
    - Download the COCO 2017 training images and annotations.
    - Organize them into the `data/train2017/` directory as expected by `utils/dataloader.py`:
      ```
      data/
      └── train2017/
          ├── annotations/
          │   └── instances_train2017.json
          └── images/
              ├── 000000000009.jpg
              └── ...
      ```
    - Update the `COCO_ROOT_DIR` variable in the notebook ([Fine*tune_SAM*(segment_anything)\_on_a_custom_dataset.ipynb](<Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb>)) if your data is stored elsewhere.

## Usage

1.  **Open the Jupyter Notebook:**
    Launch Jupyter Lab or Jupyter Notebook and open [`Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb`](<Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb>).

2.  **Configure Parameters:**

    - Adjust `class_id_to_filter` in cell `7843b87d` if you want to fine-tune on a different COCO category.
    - Modify `target_class_index` in the `EmbeddingsDataset` initialization (cell `9c59df50`) to match the class you are training for (this should be `1` if your `ground_truth_mask` in `EmbeddingsDataset` is binarized for the target class).
    - Set the number of epochs, learning rate, etc., as needed.

3.  **Run the Notebook Cells:**
    - **Data Preparation:** Loads the COCO dataset and filters for the specified class.
    - **Create Embeddings:** (Cell `cd8018f1`) This step precomputes image embeddings using SAM's vision encoder. It can be time-consuming but only needs to be run once if the `final_dataloader` doesn't change. The embeddings are saved to `precomputed_embeddings.pt`.
    - **Train the Model:** (Cell `9c59df50`) Loads the precomputed embeddings and fine-tunes the SAM decoder. Training progress and results are saved to `training_SAM_results.txt`, and the final model is saved to `final_model.pth`.
    - **Inference:** (Cell `3381567e`) Demonstrates how to load the fine-tuned model and perform inference on samples from the dataset, visualizing the results.

## Key Files

- [`Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb`](<Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb>): The main notebook containing all steps from data preparation to inference.
- [`utils/dataloader.py`](utils/dataloader.py): Contains `COCOSegmentationDataset` for loading and preprocessing COCO data and the `dataloader` function.
- [`utils/helpers.py`](utils/helpers.py): Contains the `EmbeddingsDataset` class for using precomputed embeddings.
- [`utils/metrics.py`](utils/metrics.py): Implements the `binary_dice` coefficient.
- [`models/SAM.py`](models/SAM.py): Wrapper class for the SAM model, including methods for freezing parts of the model and visualization.
- [`inference_pipeline.py`](inference_pipeline.py): Contains a `MultiModelPipeline` class, which the `SAM` class inherits from, providing a structure for model prediction.

## Notes

- The training process, especially embedding generation, can be computationally intensive and time-consuming.
- Ensure you have sufficient disk space for the COCO dataset and generated embeddings/models.
- The current setup fine-tunes for a binary segmentation task (target class vs. background).
