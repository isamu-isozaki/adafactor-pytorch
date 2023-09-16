from setuptools import setup, find_packages

setup(
  name = 'adafactor-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.0',
  license='MIT',
  description = 'adafactor Optimizer - Pytorch',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/adafactor-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
