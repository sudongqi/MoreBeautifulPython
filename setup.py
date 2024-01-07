from setuptools import setup, find_packages

setup(
    name="mbp",
    version="1.5.86",
    author="Dongqi Su",
    description="Make Python even more beautiful :) This package includes implementations that you wish were in the standard library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sudongqi/MoreBeautifulPython.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["mbp"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12.0",
    install_requires=[
        "wcwidth>=0.2.5",
        "pyyaml>=5.1",
    ],
)