from glob import glob

from setuptools import find_packages, setup

package_name = "neurogrip_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.launch.py"),
        ),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pavel",
    maintainer_email="pavelci2005@gmail.com",
    description="Launch and configuration package for NeuroGrip-X.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "set_track = neurogrip_bringup.set_track_node:main",
        ],
    },
)
