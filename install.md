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
pip install peft==0.17.1 transformers==4.44.2
```

```
pip install open3d>=0.19.0
```