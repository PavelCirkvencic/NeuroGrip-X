from setuptools import find_packages, setup

package_name = 'neurogrip_ai'

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
    description='Read-only learned-dynamics inference for NeuroGrip-X.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'contextual_inference_node = neurogrip_ai.contextual_inference_node:main',
            'inference_node = neurogrip_ai.inference_node:main',
            'physics_residual_node = neurogrip_ai.physics_residual_node:main',
            'koopman_inference_node = neurogrip_ai.physics_residual_node:main',
        ],
    },
)
