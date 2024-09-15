## Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior

> [Zhiyuan Li](https://github.com/huai-chang), Yanhui Zhou, [Hao Wei](https://github.com/cshw2021), Chenyang Ge, Jingwen Jiang<br>
> :partying_face: This work is accepted by IEEE Transactions on Circuits and Systems for Video Technology.

<p align="center">
    <img src="assets/DiffEIC.png" style="border-radius: 15px"><br>
</p>

## :book: Table Of Contents
- [:eyes: Visual Results](#visual-results)
- [:crossed\_swords: Quantitative Performance](#quantitative-performance)
- [:computer: Train](#computer-train)
- [:zap: Inference](#inference)
- [:memo: TODO](#todo)
- [:heart: Acknowledgement](#acknowledgement)
- [:clipboard: Citation](#citation)

## <a name="visual_results"></a>:eyes: Visual Results
<p align="center">
    <img src="assets/visual_results.png" style="border-radius: 15px"><br>
</p>

## <a name="quantitative_performance"></a>:crossed_swords: Quantitative Performance
<p align="center">
    <img src="assets/quantitative.png" style="border-radius: 15px"><br>
</p>

## <a name="train"></a>:computer: Train
1. Generate file list of training set and validation set.

   ```
   python3 make_fire_list.py\
   --train_folder [path_to_train_folder]\
   --test_folder [path_to_test_folder]\
   --save_folder [path_to_save_floder]
   ```
   After running this script, you will get two file lists in save_folder, each line in a file list contains an absolute path of an image file:

   ```
   save_folder
   ├── train.list # training file list
   └── valid.list # validation file list
   ```

2. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) into `./weight`.
   ```
   wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
   ```

3. Modify the configuration file `./configs/train_diffeic.yaml` and `./configs/model/diffeic.yaml` accordingly.

4. Start training.
   ```
   python3 train.py
   ```

## <a name="inference"></a>:zap: Inference
1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) into `./weight`.
   ```
   wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
   ```
 
2. Download the pre-trained weights for the LFGCM and Control Module into `./weight`.

    | Bitrate   | Link|
    | --------- | ------------------ |
    | 0.12 bpp  | [1_2_1](https://drive.google.com/drive/folders/1I_ZZZtm65aNqueXzjqpn1-ciEl_wMvCS?usp=sharing)             |
    | 0.09 bpp  | [1_2_2](https://drive.google.com/drive/folders/1I_ZZZtm65aNqueXzjqpn1-ciEl_wMvCS?usp=sharing)             |
    | 0.06 bpp  | [1_2_4](https://drive.google.com/drive/folders/1I_ZZZtm65aNqueXzjqpn1-ciEl_wMvCS?usp=sharing)              |
    | 0.04 bpp  | [1_2_8](https://drive.google.com/drive/folders/1I_ZZZtm65aNqueXzjqpn1-ciEl_wMvCS?usp=sharing)              |
    | 0.02 bpp  | [1_2_16](https://drive.google.com/drive/folders/1I_ZZZtm65aNqueXzjqpn1-ciEl_wMvCS?usp=sharing)              |

3. Run the following command.

   ```
   python3 inference_partition.py \
   --ckpt_sd ./weight/v2-1_512-ema-pruned.ckpt \
   --ckpt_lc ./weight/1_2_1/lc.ckpt \
   --config configs/model/diffeic.yaml \
   --input path to input images \
   --output path to output files \
   --steps 50 \
   --devide cuda 
   ```

## <a name="todo"></a>:memo: TODO
- [x] Release code
- [x] Release pretrained models

## <a name="acknowledgement">:heart: Acknowledgement
This work is based on [ControlNet](https://github.com/lllyasviel/ControlNet), [DiffBIR](https://github.com/XPixelGroup/DiffBIR), and [ELIC](https://github.com/JiangWeibeta/ELIC), thanks to their invaluable contributions.

## <a name="cite"></a>:clipboard: Citation

Please cite us if our work is useful for your research.

```
@article{li2024towards,
  author={Li, Zhiyuan and Zhou, Yanhui and Wei, Hao and Ge, Chenyang and Jiang, Jingwen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3455576}}
```