from glob import glob

from setuptools import find_packages, setup

package_name = "neurogrip_sim"

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
            "share/" + package_name + "/worlds",
            glob("worlds/*.sdf"),
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
    description="Gazebo simulation assets for NeuroGrip-X.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "scenario_builder = neurogrip_sim.scenario_builder:main",
        ],
    },
)
