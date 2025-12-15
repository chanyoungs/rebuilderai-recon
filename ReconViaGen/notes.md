# [ReconViaGen](https://github.com/estheryang11/ReconViaGen)
![Screenshot](outputs/screenshot.png)

## Installation notes
ReconViaGen

```
git clone --recursive https://github.com/estheryang11/ReconViaGen.git
cd ReconViaGen
```

In `requirements.txt`, remove the duplicate kornia==0.8.0 line and keep kornia==0.8.2.

```
conda create -n reconviagen python=3.10 -y
conda activate reconviagen

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu120==2.3.6 xformers==0.0.27.post2

pip install -r requirements.txt
```

Install ninja for faster build
```
pip install ninja
```

Update nvdiffrast to the latest version
```
pip uninstall nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Change peft and transformers versions
```
pip install peft==0.17.1 transformers==4.44.2 open3d
```

## Repo modifications
In order to fix the following runtime error:
```
RuntimeError: FlashAttention only support fp16 and bf16 data type
```
Add the following snippet to the end of the file `trellis/pipelines/trellis_image_to_3d.py`:
```python
# Convert models to fp16 to support FlashAttention
new_pipeline.models['sparse_structure_vggt_cond'].convert_to_fp16()
new_pipeline.models['slat_vggt_cond'].convert_to_fp16()
new_pipeline.models['sparse_structure_flow_model'].convert_to_fp16()
new_pipeline.models['slat_flow_model'].convert_to_fp16()
```

## System requirements
Required 30GB VRAM & 30GB RAM - Ended up using NVIDIA L4050 with 48GB VRAM and 128GB RAM on the cloud.

## Terminal output
<div style="height: 50vh; overflow-y: auto; border: 1px solid #ccc;">
python app_fine.py
[SPARSE] Backend: spconv, Attention: flash_attn
[SPARSE][CONV] spconv algo: native
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/gradio_client/utils.py:1097: UserWarning: file() is deprecated and will be removed in a future version. Use handle_file() instead.
  warnings.warn(
[ATTENTION] Using backend: flash_attn
/home/zeus/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:43: UserWarning: xFormers is available (SwiGLU)
  warnings.warn("xFormers is available (SwiGLU)")
/home/zeus/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:27: UserWarning: xFormers is available (Attention)
  warnings.warn("xFormers is available (Attention)")
/home/zeus/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:33: UserWarning: xFormers is available (Block)
  warnings.warn("xFormers is available (Block)")
Using cached weights/dreamsim
Using cache found in weights/dreamsim/facebookresearch_dino_main
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/peft/tuners/tuners_utils.py:196: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/gradio_client/utils.py:1097: UserWarning: file() is deprecated and will be removed in a future version. Use handle_file() instead.
  warnings.warn(
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 12.04it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:03<00:00,  3.66it/s]
Rendering: 30it [00:00, 169.78it/s]
Rendering: 1it [00:00, 203.87it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 219.01it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 223.85it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 216.12it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 189.66it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 187.29it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 268.69it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 277.60it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
/teamspace/studios/this_studio/ReconViaGen/app_fine.py:334: RuntimeWarning: Mean of empty slice.
  scale_1 = dist_1[dist_1 < np.percentile(dist_1, 99)].mean()
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/numpy/_core/_methods.py:147: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Rendering: 1it [00:00, 202.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 239.36it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 229.62it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 206.15it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 191.23it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 200.69it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 211.66it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 211.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 205.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 207.88it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]


















































100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512













































  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]






























Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]






















Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]

















 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]











 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 207.88it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]





 (Found 2 images)
Rendering: 1it [00:00, 211.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 205.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 207.88it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]

Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done 
Rendering: 30it [00:00, 117.25it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 211.66it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 211.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 205.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 207.88it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 211.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 205.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 207.88it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 218.06it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 212.08it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 155.19it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 121.47it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 196.42it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 1it [00:00, 204.02it/s]
 - adding images with resolution 518x518 --> 512x512
 - adding images with resolution 518x518 --> 512x512
 (Found 2 images)
Rendering: 6it [00:00, 207.05it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 9/12 [00:02<00:00,  3.58it/s]
Setting up [LPIPS] perceptual loss: trunk [vgg], v[0.1], spatial [off]
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to weights/dreamsim/checkpoints/vgg16-397923af.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 528M/528M [00:01<00:00, 422MB/s]
Loading model from: /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 501M/528M [00:01<00:00, 419MB/s]
Rendering: 6it [00:00, 73.07it/s]
Rendering: 6it [00:00, 224.21it/s]
Rendering: 6it [00:00, 222.51it/s]
Rendering: 6it [00:00, 230.64it/s]
Rendering: 6it [00:00, 202.51it/s]█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                    | 10/12 [00:06<00:02,  1.41s/it]
Rendering: 6it [00:00, 196.11it/s]
Rendering: 6it [00:00, 225.37it/s]
Rendering: 6it [00:00, 238.45it/s]
Rendering: 6it [00:00, 213.74it/s]
Rendering: 6it [00:00, 202.50it/s]███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                  | 11/12 [00:07<00:01,  1.38s/it]
Rendering: 6it [00:00, 216.07it/s]
Rendering: 6it [00:00, 215.87it/s]
Rendering: 6it [00:00, 214.70it/s]
Rendering: 6it [00:00, 215.33it/s]
Sampling: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:09<00:00,  1.32it/s]
Rendering: 120it [00:00, 341.33it/s]
Rendering: 120it [00:01, 117.81it/s]
INFO- Loaded 9863 vertices and 19724 faces.

100% done
Rendering: 30it [00:00, 117.25it/s]
</div>

## Output
<video autoplay loop muted playsinline>
  <source src="outputs/sample.mp4" type="video/mp4">
</video>

Click [HERE](https://playcanvas.com/model-viewer?load=https://cdn.jsdelivr.net/gh/chanyoungs/rebuilderai-recon@main/ReconViaGen/outputs/sample.glb) to see the interactive model viewer.

Click [HERE](https://cdn.jsdelivr.net/gh/chanyoungs/rebuilderai-recon@main/ReconViaGen/outputs/sample.glb) to download the 3D model.

## Dev Time
- 3 Hours