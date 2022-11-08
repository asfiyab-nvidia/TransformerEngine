Experiments in extending torchscript and ONNX with TE operators.

Use this code with Nvidia's pytorch container which contains preinstallted torch and transformer engine.

1. Build the custom-ops library

```
nzmora@fd1651e59ebd:/workspace/tmp2/te_custom_ops$ python3 setup.py install
```

This will create 3 directories: `build`, `dist`, and `custom_fp8_qdq.egg-info`.

The custom-ops lib is `build/lib.linux-x86_64-3.8/custom_fp8_qdq.cpython-38-x86_64-linux-gnu.so`


If you hit a permissions error like this:
```
[Errno 13] Permission denied: '/opt/conda/lib/python3.8/site-packages/test-easy-install-565.write-test'
```

Then you need to change permissions for the python installation dir:

```
sudo chown -R nzmora.dip /opt/conda/lib/python3.8/site-packages/
```


2. Run the test

```
python3 test_custom_onnx_export.py
```

This will generate an ONNX file.

