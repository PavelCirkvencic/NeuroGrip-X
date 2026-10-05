from setuptools import find_packages, setup

package_name = 'neurogrip_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pavel',
    maintainer_email='pavelci2005@gmail.com',
    description='Safety-guarded control, logging and simulator integration for NeuroGrip-X.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ackermann_guard = neurogrip_control.ackermann_guard:main',
            'applied_steering_bridge = neurogrip_control.applied_steering_bridge:main',
            'candidate_mux = neurogrip_control.candidate_mux:main',
            'command_guard = neurogrip_control.command_guard:main',
            'excitation_driver = neurogrip_control.excitation_driver:main',
            'eufs_excitation_driver = neurogrip_control.eufs_excitation_driver:main',
            'grip_scheduler = neurogrip_control.grip_schedule:main',
            'imu_relay = neurogrip_control.imu_relay:main',
            'matlab_tcp_bridge = neurogrip_control.matlab_tcp_bridge:main',
            'scenario_manager = neurogrip_control.scenario_manager:main',
            'state_logger = neurogrip_control.state_logger:main',
            'tracking_state_node = neurogrip_control.tracking_state_node:main',
        ],
    },
)
