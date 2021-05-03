# GateDetection
Individual assignment for course Autonomous Flight of Micro Air Vehicles 2020/2021

## Dependencies

Requirements: OpenCV, PyTorch, Jupyter Notebook

## Run
Put the dataset `WashingtonOBRace` under the repo. Download saved models: https://drive.google.com/drive/folders/1_L6Vzn01lf_hUjFGckGz6qNewVFDrCHq?usp=sharing. Then put them in the folder `learned_weights/` in the repo root directory. 

The codes is with the same structure of the report. For only checking the proposed method(report section 3), just run `gate_detection.ipynb`. 

- Report section 2.1

   - `gate_detection_orb.ipynb` : explore the ORB features extraction method(if it works, filter good matches, visualize, etc)
    -  `gate_detection_orb_test.ipynb`:  finally test the ORB features method(computational effort, ROC curve)
- Report section 2.2
  - `gate_detection_cnn.ipynb`: explore the binary classification method. **Skip** the node for training the model, and later in the code *'learned_weights/gate_det_cnn.pt'* is loaded for testing. Actually for avoiding accidentally starting training I have commented them.
- Report section 2.3
  - `gate_detection_maskrcnn.ipynb`: explore the Mask R-CNN method. **Skip** the node for training the model, and later in the code *'learned_weights/gate_det_maskrcnn'* is loaded for testing
- Report section 3
  - `gate_detection.ipynb`:  test the final proposed method. Run the notebook step by step