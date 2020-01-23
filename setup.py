import setuptools

with open("README.rst") as fp:
    long_description = fp.read()

with open("torchutils/_version.py") as fp:
    torchutils_version = fp.read().strip().split('__version__ = ')[1][1:-1]

setuptools.setup(
    name='torchutils', version=torchutils_version,
    author="Anjandeep Singh Sahni", author_email="sahni.anjandeep@gmail.com",
    description="PyTorch utility APIs.", long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/anjandeepsahni/torchutils.git",
    packages=setuptools.find_packages(exclude=['test']), classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], keywords='machine-learning deep-learning pytorch neuralnetwork',
    install_requires=['torch>=1.0.0', 'numpy>=1.16.2',
                      'matplotlib>=3.0.3'], license='MIT')
