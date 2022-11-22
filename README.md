## Export Transformer Engine Ops to ONNX

The below instructions are for the following environment:
1. Hopper environment obtained using crun. Additional mounting of scratch space is required
2. Launch the latest NVIDIA PyTorch docker container

Command to run from computelab-304 
```bash
# enter scratch space
cd /home/scratch.asfiyab_sw/

# request hopper machine
/home/scratch.svc_compute_arch/release/crun/latest/crun/crun -q 'gpu_arch=Hopper' -i -img nvcr.io/nvidia/pytorch:22.09-py3 --args="-v $PWD:/scratch_space" -t 04:00:00
```

### Build Transformer Engine Extensions with Custom TorchScript operator

#### Enable write permissions

```bash
sudo chown -R asfiyab.dip /opt/conda/lib/python3.8/site-packages/
```

#### Build TE
To avoid running into space issues, perform the installation within your scratch space. To do this, create a build directory as done below and pass it to the `--target` flag during the pip install.
```bash
mkdir build
pip install -e . --target=build/ --upgrade

export PYTHONPATH=$PYTHONPATH:$PWD

# verify pip sets TE directory to PWD
pip show transformer_engine
```

The installed shared library with the TS op is located at: 
```bash
./build/transformer_engine_extensions.cpython-38-x86_64-linux-gnu.so
```

However, we do not need to load it explicitly anymore. Building it with the TE extensions enables us to load the transformer engine extensions and call it using `torch.ops.<namespace>.<function_name>`

### Run the ONNX export script

Usage:

```bash
# test fp8 path
python onnx_export/test_te_gemm_export.py --fp8

# test non-fp8 path with bias
python onnx_export/test_te_gemm_export.py

# test non-fp8 path with gelu
python onnx_export/test_te_gemm_export.py --test_gelu
```