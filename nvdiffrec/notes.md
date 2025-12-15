# nvdiffrec

## Installation notes
```bash
conda create -n dmodel python=3.10 -y
activate dmodel
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge -y
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

### If tiny-cuda-nn installation fails (C++17 issue on RTX 3060), do the following steps:

1. ```git clone --recursive https://github.com/NVlabs/tiny-cuda-nn```
1. Patch setup.py (Force C++17) Run this command to replace the 14 with 17 in the bindings setup file you just shared:

    ```bash
    sed -i 's/cpp_standard = 14/cpp_standard = 17/g' bindings/torch/setup.py
    ```

1. Patch rtc_kernel.cu (Force C++17 for Runtime) This ensures the JIT compiler doesn't crash later when you run your code.

    ```bash
    sed -i 's/std=c++14/std=c++17/g' src/rtc_kernel.cu
    ```

1. Install Navigate to the bindings folder and install. (Note: We use pip install . to install the patched local version)

    ```bash
    cd bindings/torch

    # Prevent RAM crash
    export MAX_JOBS=1

    # Target your RTX 3060 specifically
    export TCNN_CUDA_ARCHITECTURES=86

    # Install!
    pip install . --no-build-isolation
    ```

## Config file
```json
{
    "ref_mesh": "data/shoe",
    "random_textures": true,
    "iter": 5000,
    "save_interval": 100,
    "texture_res": [1024, 1024],
    "train_res": [1024, 1024],
    "batch": 2,
    "learning_rate": [0.03, 0.01],
    "ks_min" : [0, 0.08, 0.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 2.1,
    "laplace_scale" : 3000,
    "display": [{"latlong" : true}, {"bsdf" : "kd"}, {"bsdf" : "ks"}, {"bsdf" : "normal"}],
    "background" : "white",
    "out_dir": "shoe_5000iter_d3_1024"
}
```

## transforms.json
```json
{
    "camera_angle_x": 0.85,
    "frames": [
        {
            "file_path": "./train/front",
            "transform_matrix": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 3],
                [0, 0, 0, 1]
            ]
        },
        {
            "file_path": "./train/back",
            "transform_matrix": [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, -3],
                [0, 0, 0, 1]
            ]
        },
        {
            "file_path": "./train/left",
            "transform_matrix": [
                [0, 0, 1, 3],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ]
        },
        {
            "file_path": "./train/right",
            "transform_matrix": [
                [0, 0, -1, -3],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ]
        },
        {
            "file_path": "./train/top",
            "transform_matrix": [
                [-1, 0, 0, 0],
                [0, 0, 1, 3],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]
        },
        {
            "file_path": "./train/bottom",
            "transform_matrix": [
                [1, 0, 0, 0],
                [0, 0, -1, -3],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]
        }
    ]
}
```

- camera_angle_x: Field of view in radians. A typical FOV with mild distortion is around 49 degrees (0.85 radians).
- transform_matrix: 4x4 transformation matrix for camera pose

# Time
Start: 2025/12/11 13:35
Pause: 2025/12/11 18:47
Start: 2025/12/11 23:40

46 back
71 front
59 right
20 left