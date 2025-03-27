from setuptools import find_packages, setup

setup(
    name="schemach_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)