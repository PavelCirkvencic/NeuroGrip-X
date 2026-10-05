from glob import glob

from setuptools import find_packages, setup

package_name = "neurogrip_visualization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pavel",
    maintainer_email="pavelci2005@gmail.com",
    description="Gazebo Fortress visual frontend and odometry sync for NeuroGrip-X.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "visual_sync_node = neurogrip_visualization.visual_sync_node:main",
            "video_recorder_node = neurogrip_visualization.video_recorder_node:main",
            "build_visual_world = neurogrip_visualization.build_world:main",
        ],
    },
)
