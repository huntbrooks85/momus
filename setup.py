# setup.py
from setuptools import setup, find_packages

setup(
    name="momus",
    version="0.1.0",
    license="MIT License",
    packages=find_packages(),
    install_requires=[
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "astropy",
            "astroquery",
            "photutils",
            "scikit-learn",
            "xarray",
            "tqdm",
            "h5py",
            "emcee",
            "corner",
            "pathos",
            "requests", 
            "pyvo"
                    ],
    author="Hunter Brooks",
    author_email="hbrooks8@rockets.utoledo.edu",
    description="Intrinsic Scatter Regressions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huntbrooks85/momus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
