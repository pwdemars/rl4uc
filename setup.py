import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkgs = {
    "required": [
        "numpy",
        "pandas",
    ]
}

setuptools.setup(
    name="rl4uc", 
    version="0.0.3",
    author="Patrick de Mars",
    author_email="pwdemars@gmail.com",
    description="Reinforcement learning environment for the unit commitment problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=pkgs["required"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'rl4uc': ['data/*.csv']},
)
