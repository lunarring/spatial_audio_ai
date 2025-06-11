from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatial-audio-ai",
    version="0.1.0",
    author="Lunar Ring",
    description=(
        "A toolkit for audio processing, spatial sound control, "
        "and diffusion-based sound generation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lunarring/spatial_audio_ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "sounddevice",
        "soundfile",
        "diffusers",
        "numpysocket",
    ],
    dependency_links=[
        "git+https://github.com/lunarring/lunar_tools"
    ],
    entry_points={
        'console_scripts': [
            'bh=spatial_audio_ai.tools.client:main',
        ],
    },
) 