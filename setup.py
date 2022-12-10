from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("betty/version.txt", "r") as fv:
    version = fv.read()

with open("requirements/requirements.txt", "r") as f:
    requirements = f.read().splitlines()

description = (
    "An automatic differentiation library for multilevel optimization and "
    "generalized meta-learning"
)

python_requires = ">=3.6.0,<3.11.0"

# run setup
setup(
    name="betty",
    version=version,
    author="Sang Choe",
    author_email="sangkeun00@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leopard-ai/betty",
    keywords=[
        "meta-learning",
        "pytorch",
        "multilevel optimization",
        "machine learning",
        "artificial intelligence",
    ],
    packages=find_packages(
        exclude=[
            "examples",
            "docs",
            "tests",
            "tutorials",
        ]
    ),
    install_requires=requirements,
    license="Apache",
    python_requires=python_requires,
    # Not sure
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
