from setuptools import find_packages, setup

package_name = 'connect_myo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml',
                                   'launch/myo_data.launch.xml',
                                   ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jialuyu',
    maintainer_email='jialuyu2024@u.northwestern.edu',
    description='Myo band data',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_myo_node = connect_myo.RL_myo_node:main',
            'ru_myo_node = connect_myo.RU_myo_node:main',
            'connect_manager = connect_myo.Connection_manager:main',
        ],
    },
)
