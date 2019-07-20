import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
     name='torchutils',
     version='0.0.1',
     author="Anjandeep Singh Sahni",
     author_email="sahni.anjandeep@gmail.com",
     description="PyTorch utility functions.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/anjandeepsahni/pytorch_utils",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     keywords='machine-learning deep-learning pytorch neuralnetwork',
     license='MIT'
 )
