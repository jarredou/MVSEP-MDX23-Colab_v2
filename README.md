# MVSep-MDX23 Colab Fork v2.2.2
Adaptation of MVSep-MDX23 algorithm for Colab, with few tweaks:

v2.3 is now available: https://github.com/jarredou/MVSEP-MDX23-Colab_v2/

**v2.2.2**
* D1581 demo model replaced by new InstVocHQ MDXv3 model.

**v2.2.1**
* Added custom weights feature
* Fixed some bugs
* Fixed input: you can use a file or a folder as input now

**v2.2**
* Added MDXv3 compatibility
* Added MDXv3 demo model D1581 in vocals stem multiband ensemble
* Added MDX-VOC-FT *Fullband SRS* in vocals stem multiband ensemble
* Added option to output only vocals/instrum stems (faster processing)
* Added 16bit output format option
* Added "BigShift trick" for MDX models
* Added separated overlap values for MDX, MDXv3 and Demucs
* Fixed volume compensations fine-tuned
* Fixed some memory issues

[**v2.1 (by deton24)**](https://github.com/deton24/MVSEP-MDX23-Colab_v2.1)
* Updated with MDX-VOC-FT instead of Kim Vocal 2

[**v2.0**](https://github.com/jarredou/MVSEP-MDX23-Colab_v2/tree/2.0)
* Updated with new Kim Vocal 2 & UVR-MDX-Instr-HQ3 models
* Folder batch processing
* Fixed high frequency bleed in vocals
* Fixed volume compensation for MDX models

https://colab.research.google.com/github/jarredou/MVSEP-MDX23-Colab_v2/blob/v2.2/MVSep-MDX23-Colab.ipynb


---
Original work =>
---
# MVSEP-MDX23-music-separation-model
Model for [Sound demixing challenge 2023: Music Demixing Track - MDX'23](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023). Model perform separation of music into 4 stems "bass", "drums", "vocals", "other". Model won 3rd place in challenge (Leaderboard C).

Model based on [Demucs4](https://github.com/facebookresearch/demucs), [MDX](https://github.com/kuielab/mdx-net) neural net architectures and some MDX weights from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project (thanks [Kimberley Jensen](https://github.com/KimberleyJensen) for great high quality vocal models). Brought to you by [MVSep.com](https://mvsep.com).
## Usage

```
    python inference.py --input_audio mixture1.wav mixture2.wav --output_folder ./results/
```

With this command audios with names "mixture1.wav" and "mixture2.wav" will be processed and results will be stored in `./results/` folder in WAV format.

* **Note 1**: If you have not enough GPU memory you can use CPU (`--cpu`), but it will be slow. Additionally you can use single ONNX (`--single_onnx`), but it will decrease quality a little bit. Also reduce of chunk size can help (`--chunk_size 200000`).
* **Note 2**: In current revision code requires less GPU memory, but it process multiple files slower. If you want old fast method use argument `--large_gpu`. It will require > 11 GB of GPU memory, but will work faster.  

## Quality comparison

Quality comparison with best separation models performed on [MultiSong Dataset](https://mvsep.com/quality_checker/leaderboard2.php?sort=bass). 

| Algorithm     | SDR bass  | SDR drums  | SDR other  | SDR vocals  | SDR instrumental  |
| ------------- |:---------:|:----------:|:----------:|:----------:|:------------------:|
| MVSEP MDX23   | 12.5034   | 11.6870    | 6.5378     |  9.5138    | 15.8213            |
| Demucs HT 4   | 12.1006   | 11.3037    | 5.7728     |  8.3555    | 13.9902            |
| Demucs 3      | 10.6947   | 10.2744    | 5.3580     |  8.1335    | 14.4409            |
| MDX B         | ---       | ----       | ---        |  8.5118    | 14.8192            |

* Note: SDR - signal to distortion ratio. Larger is better.

## GUI

![GUI Window](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/blob/main/images/MVSep-Window.png)

* Script for GUI (based on PyQt5): [gui.py](gui.py).
* You can download standalone program for Windows here: [zip1](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/releases/download/v1.0/MVSep-MDX23.zip.001), [zip2](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/releases/download/v1.0/MVSep-MDX23.zip.002). Unzip archives and to start program double click `run.bat`.
* Program will download all needed neural net models from internet at the first run.
* GUI supports Drag & Drop of multiple files.
* Progress bar available.

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
