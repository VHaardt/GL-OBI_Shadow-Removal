# Global and Local Re-Illumination Methods for Efficient Single Image Shadow Removal
This is the official implementation of the thesis [Global and Local Re-Illumination Methods for Efficient Single Image Shadow Removal](-).

## Introduction
To tackle image shadow removal problem, we propose three novel re-illumination-based methods for single-image shadow removal: G-OI-Net, L-OBI-Net, and G4L-OBI-Net. Being re-illumination-based these methods focus on re-illumination parameter estimation rather than generating entirely new pixel values, which simplifies the overall system and reduces the risk of introducing artifacts. These convolutional neural network frameworks generate overexposed images to effectively address shadowed regions, leveraging the shadow boundary region (penumbra) as a key focus area due to its complexity. The proposed methods restore shadowed areas by inpainting overex-posed regions with the original shadowed image, ensuring the preservation of non-shadowed areas. 

The models operate at different levels:

- **G-OI-Net** is a fast and efficient shadow removal solution. This method is designed to extract three global re-illumination parameters, one for each RGB channel, optimized to achieve the best overexposure effect. The overexposed image, combined with an inpainting process in the shadow-affected areas, yields the final shadow-free result. Furthermore, G-OI-Net introduced a pre-overexposure stage, where the input shadow region
is normalized and re-illuminated using parameters derived from the shadow contour region.

<p align=center><img width="80%" src="doc/pipeline.jpg"/></p>

- **L-OBI-Net**, is a more complex U-Net-based solution aimed at achieving optimal shadow removal while maintaining efficiency. This method integrates boundary awareness, which significantly improves shadow removal results and reduces ghosting effects. In addition to the shadow-affected image and corresponding shadow mask, the model also requires a penumbra region mask as input, enabling a focus on this critical transition area. This focus is further reinforced in the loss function. The network predicts pixel-wise re-illumination kernels, applied to the shadow image to generate the overexposed image. The final shadow-free image is obtained by merging the original non-shadow regions with the over-
exposed shadow regions through an inpainting process. This ensures smooth transitions in the penumbra region, achieving effective shadow removal.

<p align=center><img width="80%" src="doc/pipeline.jpg"/></p>

- **G4L-OBI-Net**, was conceptualized to implement a global step as a preliminary stage for local refinement. However, this solution did not achieve the desired results. In this method, the input shadow image undergoes a global overexposure step as an initial approximation of shadow removal, brightening the shadowed areas uniformly. This is followed by a pixel-wise local refinement stage, where each pixel in the shadow region is adjusted using locally derived parameters.

<p align=center><img width="80%" src="doc/pipeline.jpg"/></p>

Additionally, we identified and critiqued a common issue in evaluating shadow removal performance across different regions of an image (shadow region, non-shadow region, and entire image). We highlighted how the conventional application of shadow masks distorts results, making it impossible to compare performance across different areas of the same image or dataset. We proposed a solution that evaluates only the relevant pixels or areas, yielding meaningful numerical results that enable a fair comparison of how shadow removal methods perform in shadow and non-shadow regions.

For more details, please refer to our [original work](-)

## Requirement
* Python 3.7
* Pytorch 1.7
* CUDA 11.1
```bash
pip install -r requirements.txt
```

## Datasets
* ISTD [[link]](https://github.com/DeepInsight-PCALab/ST-CGAN)  
* ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)
* SRD [[Training]](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view)[[Testing]](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view)

## Pretrained models
[ISTD](https://drive.google.com/file/d/1bHbkHxY5D5905BMw2jzvkzgXsFPKzSq4/view?usp=share_link) | [ISTD+](https://drive.google.com/file/d/10pBsJenoWGriZ9kjWOcE4l4Kzg-F1TFd/view?usp=share_link) | [SRD]()

Please download the corresponding pretrained model and modify the `weights` in `test.py`.

## Test
You can directly test the performance of the pre-trained model as follows
1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` 
```python
input_dir # shadow image input path -- Line 27
weights # pretrained model path -- Line 31
```
2. Test the model
```python
python test.py --save_images
```
You can check the output in `./results`.

## Train
1. Download datasets and set the following structure
```
|-- ISTD_Dataset
    |-- train
        |-- train_A # shadow image
        |-- train_B # shadow mask
        |-- train_C # shadow-free GT
    |-- test
        |-- test_A # shadow image
        |-- test_B # shadow mask
        |-- test_C # shadow-free GT
```
2. You need to modify the following terms in `option.py`
```python
train_dir  # training set path
val_dir   # testing set path
gpu: 0 # Our model can be trained using a single RTX A5000 GPU. You can also train the model using multiple GPUs by adding more GPU ids in it.
```
3. Train the network
If you want to train the network on 256X256 images:
```python
python train.py --warmup --win_size 8 --train_ps 256
```
or you want to train on original resolution, e.g., 480X640 for ISTD:
```python
python train.py --warmup --win_size 10 --train_ps 320
```

## Evaluation
The results reported in the paper are calculated by the `matlab` script. Details refer to `evaluation/measure_shadow.m`. ...

## Results
#### Evaluation on ISTD+ with 
The evaluation results on ISTD are as follows
| Dataset         | Method                     | Params | PSNR (S) ↑ | SSIM (S) ↑ | MAE (S) ↓ | PSNR (NS) ↑ | SSIM (NS) ↑ | MAE (NS) ↓ | PSNR (ALL) ↑ | SSIM (ALL) ↑ | MAE (ALL) ↓ |
|-----------------|----------------------------|--------|------------|------------|------------|--------------|--------------|--------------|---------------|---------------|--------------|
| **256 × 256**  | Input Image                | -      | 20.83      | 0.9266     | 36.95      | 37.46        | 0.9846       | 2.42         | 20.46         | 0.8940        | 8.40         |
|                 | SP+M-Net | 5.7M   | 37.59      | 0.9899     | 6.32       | 36.02        | 0.9757       | 2.95         | 32.94         | 0.9617        | 3.46         |
|                 | DHAN | 21.8M  | 32.91      | 0.9876     | 9.57       | 27.15        | 0.9714       | 7.41         | 25.66         | 0.9561        | 7.77         |
|                 | Fu et al. | 186.5M | 36.04      | 0.9782     | 6.69       | 31.16        | 0.8922       | 3.77         | 29.45         | 0.8612        | 4.23         |
|                 | SG-ShadowNet | 4.6M   | 36.80      | 0.9901     | 6.45       | 35.57        | 0.9777       | 2.89         | 32.46         | 0.9616        | 3.41         |
|                 | BMNet | 4.4M   | 37.87      | 0.9912     | 5.83       | 37.51        | 0.9850       | 2.45         | 33.98         | 0.9722        | 2.97         |
|                 | ShadowFormer | 9.3M   | 39.48      | 0.9915     | 5.32       | 38.82        | 0.9825       | 2.30         | 35.46         | 0.9711        | 2.78         |
|                 | DeS3   | -      | 36.49      | 0.9892     | 6.54       | 34.72        | 0.9723       | 3.30         | 31.39         | 0.9573        | 3.86         |
|                 | LFG-Diffusion | -      | 39.19      | 0.9923     | 5.10       | 37.83        | 0.9842       | 2.47         | 34.76         | 0.9738        | 2.87         |
|                 | **G-OI-Net**               | 23.5M  | 34.90      | 0.9815     | 9.97       | 37.27        | 0.9844       | 2.42         | 32.35         | 0.9537        | 3.51         |
|                 | **L-OBI-Net**              | 46.8M  | 37.46      | 0.9905     | 6.02       | 37.69        | 0.9845       | 2.41         | 33.90         | 0.9708        | 2.97         |
|                 | **G4L-OBI-Net**            | 70.3M  | 36.62      | 0.9886     | 7.03       | 37.63        | 0.9846       | 2.41         | 33.40         | 0.9680        | 3.12         |
| **Original**    | Input Image                | -      | 20.81      | 0.9266     | 38.53      | 33.88        | 0.9553       | 3.33         | 20.22         | 0.8748        | 3.05         |
|                 | DHAN | 21.8M  | 32.45      | 0.9834     | 10.17      | 26.21        | 0.9433       | 7.90         | 24.86         | 0.9244        | 2.75         |
|                 | SG-ShadowNet | 4.6M   | 35.96      | 0.9845     | 7.27       | 32.76        | 0.9503       | 3.79         | 30.50         | 0.9305        | 1.43         |
|                 | BMNet | 4.4M   | 36.81      | 0.9865     | 6.58       | 34.47        | 0.9577       | 3.24         | 31.85         | 0.9411        | 1.24         |
|                 | ShadowFormer | 9.3M   | 38.07      | 0.9864     | 6.16       | 35.15        | 0.9550       | 3.15         | 32.78         | 0.9388        | 1.20         |
|                 | LFG-Diffusion | -      | 37.74      | 0.9865     | 6.07       | 34.30        | 0.9515       | 3.51         | 32.11         | 0.9357        | 1.30         |
|                 | **G-OI-Net**               | 23.5M  | 33.76      | 0.9747     | 11.13      | 33.81        | 0.9552       | 3.34         | 30.23         | 0.9199        | 1.46         |
|                 | **L-OBI-Net**              | 46.8M  | 36.48      | 0.9854     | 6.90       | 34.42        | 0.9557       | 3.30         | 31.69         | 0.9380        | 1.28         |
|                 | **G4L-OBI-Net**            | 70.3M  | 35.60      | 0.9817     | 7.84       | 34.35        | 0.9558       | 3.30         | 31.29         | 0.9335        | 1.32         |


#### Visual Results
<p align=center><img width="80%" src="doc/res.jpg"/></p>

#### Testing results
The testing results on dataset ISTD, ISTD+, SRD are: [results](https://drive.google.com/file/d/1zcv7KBCIKgk-CGQJCWnM2YAKcSAj8Sc4/view?usp=share_link)

## References
Our implementation is based on [Uformer](https://github.com/ZhendongWang6/Uformer) and [Restormer](https://github.com/swz30/Restormer). We would like to thank them.

Citation
-----
Preprint available [here](https://arxiv.org/pdf/2302.01650.pdf). 

In case of use, please cite our publication:

L. Guo, S. Huang, D. Liu, H. Cheng and B. Wen, "ShadowFormer: Global Context Helps Image Shadow Removal," AAAI 2023.

Bibtex:
```
@article{guo2023shadowformer,
  title={ShadowFormer: Global Context Helps Image Shadow Removal},
  author={Guo, Lanqing and Huang, Siyu and Liu, Ding and Cheng, Hao and Wen, Bihan},
  journal={arXiv preprint arXiv:2302.01650},
  year={2023}
}
```

## Contact
If you have any questions, please contact lanqing001@e.ntu.edu.sg
