import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='aetk',
    version='1.0',
    url='https://github.com/sudongqi/AbsolutelyEssentialToolKit.git',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author='Dongqi Su',
    description='light-weight essential tools that are not included in Python standard library',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'tqdm>=4.64.0'
    ],
)