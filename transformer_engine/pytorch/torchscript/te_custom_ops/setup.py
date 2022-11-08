from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(
    name='te_torchscript_ext',

    ext_modules=[
        cpp_extension.CppExtension(
            name='te_torchscript_ext',
            sources=['custom_fp8_ops.cpp'],
            extra_compile_args=['-g'],
        )
    ],

    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
