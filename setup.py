import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tardigrad",
    version="0.1.0",
    author="Serif Agir",
    author_email="serifagir64@gmail.com",
    description="Tiny and lightweight auto gradient engine implements backward propagation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serifagir/tardigrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=False,
)
