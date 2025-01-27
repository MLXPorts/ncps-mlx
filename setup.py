# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ncps-mlx",
    version="1.0.0",
    packages=find_packages(),  # include/exclude arguments take * as wildcard, . for any sub-package names
    description="Neural Circuit Policies for Apple MLX (fork)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sydneypdx/ncps-hpc-limb",
    author="Sydney Renee",
    author_email="sydney@solace.ofharmony.ai",
    license="Apache License 2.0",
    # tensorflow and torch isn't a dependency because it would force the
    # download of the gpu version or the cpu version.
    # users should install it manually.
    install_requires=[
        "packaging>=20.0",
        "future>=0.18.2",
        "mlx>=0.1.0"
        # "networkx",
        # "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx_rtd_theme>=0.5.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
