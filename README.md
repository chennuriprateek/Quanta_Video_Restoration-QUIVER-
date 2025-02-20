<div align="center">

# 【ECCV'2024🔥】Quanta Video Restoration
</div>

## Overview
QUIVER (Quanta Video Restoration) is a deep learning-based framework for restoring quanta video data. The project focuses on enhancing extreme low-light and high-speed imaging through advanced post-processing techniques.


## [🔥 Project Page](https://chennuriprateek.github.io/Quanta_Video_Restoration-QUIVER-/) | [Paper](https://arxiv.org/pdf/2410.14994) 

## Features
- Model checkpoints trained on simulated data
- Optical flow extraction and processing
- Patch-based training for efficient learning

## 🧩 Dataset and Pre-train Models
<table>
  <tr>
    <th rowspan="2">Datasets</th>
    <th rowspan="2">Model</th>
    <th colspan="4">Pre-trained Models</th>
  </tr>
  <tr>
    <th>3.25 PPP</th>
    <th>9.75 PPP</th>
    <th>19.5 PPP</th>
    <th>26 PPP</th>
  </tr>
  <tr>
    <td rowspan="6"><a href="https://app.box.com/s/0yzzajq1pnhyya057ilerzjia4qtsvhc">I2_2000FPS</a></td>
    <th>QUIVER</th>
    <td><a href="https://app.box.com/s/bq1rsi6zb4k8scbpnxz91x459zuno8dj">Link</a></td>
    <td><a href="https://app.box.com/s/1oniapgejj7tuttto26n7bfj6755kql3">Link</a></td>
    <td><a href="https://app.box.com/s/6hhvrn7wc91d8izfeet3r9qa753sazz4">Link</a></td>
    <td><a href="https://app.box.com/s/fwd2kw7mjs3fb751yfszz4giami3xaa1">Link</a></td>
  </tr>
  <tr>
    <th>EMVD</th>
    <td><a href="https://app.box.com/s/gyvjq2192vd2cizczf2tdq18jqg2poy3">Link</a></td>
    <td><a href="https://app.box.com/s/7tir7ubgym8qp6k64omq1xb2nesmebwl">Link</a></td>
    <td><a href="https://app.box.com/s/5xikwq9tgb37rl7b3n52tubnchrmbp3l">Link</a></td>
    <td><a href="https://app.box.com/s/wjs5r5v2lardasfoiense3zljpex9fss">Link</a></td>
  </tr>
  <tr>
    <th>Spk2ImgNet</th>
    <td><a href="https://app.box.com/s/x2lzjj2vwqszyfo9xi8cn9h9j6wtsqrh">Link</a></td>
    <td><a href="https://app.box.com/s/0c5gwot54fvxktqg64llr7i0b26ktvxa">Link</a></td>
    <td><a href="https://app.box.com/s/5nlcacm4cuswttwi5i9qqy2lxg4gkyhr">Link</a></td>
    <td><a href="https://app.box.com/s/b22lapr1acz21q4xttmnjb6rd0b0cgag">Link</a></td>
  </tr>
  <tr>
    <th>FloRNN</th>
    <td><a href="https://app.box.com/s/045lpgtqhlhtgedv87ulzviko9bskj03">Link</a></td>
    <td><a href="https://app.box.com/s/alrx2ezke493oz47idv8slrw7pg3jw6g">Link</a></td>
    <td><a href="https://app.box.com/s/679lfj2pdzyb3ot9p0jrlcc3tgaomp7e">Link</a></td>
    <td><a href="https://app.box.com/s/p2hqgxv2np40whlx0t4k66n0z5wlom7j">Link</a></td>
  </tr>
  <tr>
    <th>RVRT</th>
    <td><a href="https://app.box.com/s/tobnk3l2xdye7qoehp61n9mly0oam92f">Link</a></td>
    <td><a href="https://app.box.com/s/35told3q0nm45ykn5ggzngni7xxms6qr">Link</a></td>
    <td><a href="https://app.box.com/s/hautpgtjztaj854t2xcpuwlx32ou77fs">Link</a></td>
    <td><a href="https://app.box.com/s/l1pwy8yn0semjbssxdl18jbolx8uqlbl">Link</a></td>
  </tr>
</table>

## Installation
### Creating the Conda Environment
To set up the environment, run the following command:
```sh
conda env create -f QUIVER_environment.yml
```

## Model Checkpoints
- All model checkpoints are trained on simulated data.
- Checkpoints are named using the following convention:
  ```
  [model_name]_[past_frames(p)][#past_frames]_[future_frames(f)][#future_frames]_[#photons_per_pixel_per_frame]PPP.pth
  ```
  - **Total input frames**: `#past_frames + #future_frames + 1`

## Code Structure
The repository is organized into the following scripts:

### 1. `dataloader.py`
- Loads data directly from MP4 videos.
- Contains details on the simulation used for training.

### 2. `model.py` & `archs.py`
- Defines the model architecture and related modules.

### 3. `input_args.py`
- Contains hyperparameters controlling training and testing:
  - `num_frames`: Number of frames used as input to the model.
  - `patch_size`: Patch size for training.
  - `future_frames`: Number of frames to the right of the reference frame.
  - `past_frames`: Number of frames to the left of the reference frame.
  - `weights_dir`: Checkpoint path.
  - `load_model_flag`: Boolean flag to load model checkpoints.
  - `lr`: Learning rate.
  - `batch_size`: Batch size during training (default batch size for testing is 1).
  - `save_path`: Directory to save outputs during testing.

### 4. `test.py`
- Modify the input hyperparameters mentioned above and run the following command for testing:
  ```sh
  python test.py
  ```

### 5. `train.py`
- Modify the input hyperparameters mentioned above and run the following command for training:
  ```sh
  python train.py
  ```

## Post-Processing
- After generating outputs, post-processing is done using MATLAB’s `localtonemap` function.
- The same post-processing is applied to all models including the baselines.

## Contact
For any questions or clarifications, feel free to reach out.

---

This README provides a structured guide to installing, training, and testing QUIVER. Let me know if you'd like any modifications!





## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{chennuri2025quanta,
  author={Chennuri, Prateek and Chi, Yiheng and Jiang, Enze and Godaliyadda, G.M. and Gnanasambandam, Abhiram and Sheikh, Hamid R. and Gyongy, Istvan and Chan, Stanley H.},
  title={Quanta Video Restoration},
  booktitle={European Conference on Computer Vision},
  pages={152--171},
  year={2025},
  organization={Springer}
}
```
<!---               
## 🔑 Setup and Prepare LMDB files
```
this is a code place holder
```
place holder

## 🛠️ Training
place holder

## 🚀 Performance Evaluation
place holder

## 👍 Useful Links
place holder, put few other datasets here

## 📜 Citation
place holder
``` -->
