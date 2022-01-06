from __future__ import absolute_import
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'noise2sim','version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()


setup(name='noise2sim',
      version=__version__,
      description='Noise2Sim suppresses both independent and correlated nosies through training a neural network'
                  'in a self-supervised learning manner.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/niuchuangnn/noise2sim',
      author='Chuang Niu',
      author_email='niuchuangnn@gmail.com',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Repository': 'https://github.com/niuchuangnn/noise2sim',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.6',
      ],

      scripts=['scripts/train.py',
      ],

      install_requires=[
          "lmdb",
          "tqdm",
          "imageio",
          "addict",
          "opencv-python",
          "matplotlib",
          "pydicom",
          "scipy",
          "faiss-gpu",
          "scikit-image",
        ]
      )