# <img src="assets/badges/lotus_icon.png" alt="lotus" style="height:1em; vertical-align:bottom;"/> Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://lotus-2.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2409.18124)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20(Depth)-yellow)](https://huggingface.co/spaces/haodongli/Lotus-2_Depth)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20(Normal)-yellow)](https://huggingface.co/spaces/haodongli/Lotus-2_Normal)

[Jing He](https://scholar.google.com/citations?hl=en&user=RsLS11MAAAAJ)<sup>1</sup>,
[Haodong Li](https://haodong-li.com/)<sup>12<span>&#10033;</span></sup>,
[Mingzhi Sheng]()<sup>1<span>&#10033;</span></sup>,
[Ying-Cong Chen](https://www.yingcong.me/)<sup>13&#9993;</sup>

<span class="author-block"><sup>1</sup>HKUST(GZ)</span>
<span class="author-block"><sup>2</sup>UC San Diego</span>
<span class="author-block"><sup>3</sup>HKUST</span><br>
<span class="author-block">
    <sup>&#10033;</sup>Both authors contributed equally.
    <sup>&#9993;</sup>Corresponding author.
</span>

![teaser](assets/badges/teaser-1.png)

**We present Lotus-2, a two-stage deterministic framework for monocular geometric dense prediction.** Our method leverages pre-trained generative model as a deterministic world prior to achieve **new state-of-the-art accuracy** while requiring **remarkably minimal data** (trained on only **0.66%** of the samples used by MoGe-2). This figure demonstrates Lotus-2's robust zero-shot generalization with sharp geometric details, especially in challenging cases like oil paintings and transparent objects.

🚀🚀🚀 **Please also check the** [**Project Page**](https://lotus3d.github.io/) **and** [**Github Repo**](https://github.com/EnVision-Research/Lotus) **our prior work: Lotus!** 🚀🚀🚀

## 📢 News
- 2025-11-29: The inference code and HuggingFace demo ([Depth](https://huggingface.co/spaces/haodongli/Lotus-2_Depth) & [Normal](https://huggingface.co/spaces/haodongli/Lotus-2_Normal)) are available! <br>
- 2025-11-29: [Paper](https://arxiv.org/abs/2409.18124) released. <br>

## 🛠️ Setup
This installation was tested on: Ubuntu 20.04 LTS, Python 3.10, CUDA 12.3, NVIDIA A800-SXM4-80GB.  

1. Clone the repository (requires git):
```
git clone https://github.com/EnVision-Research/Lotus-2.git
cd Lotus-2
```

2. Install dependencies (requires conda):
```
conda create -n lotus2 python=3.10 -y
conda activate lotus2
pip install -r requirements.txt 
```

3. Be sure you have access to [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev).

4. Login your huggingface account via:
```
hf auth login
```

## 🤗 Gradio Demo

1. Online demo: [Depth](https://huggingface.co/spaces/haodongli/Lotus-2_Depth) & [Normal](https://huggingface.co/spaces/haodongli/Lotus-2_Normal)
2. Local demo
- For **depth** estimation, run:
    ```
    python app.py depth
    ```
- For **normal** estimation, run:
    ```
    python app.py normal
    ```

## 🕹️ Inference
### Testing on your images
1. Place your images in a directory, for example, under `./assets/in-the-wild_example` (where we have prepared several examples). 
2. Run the inference command: `sh infer.sh`. 

## 🎓 Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
Comming soon!
```
