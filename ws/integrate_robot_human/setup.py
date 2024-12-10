from setuptools import find_packages, setup

package_name = 'integrate_robot_human'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jialuyu',
    maintainer_email='jialuyu2024@u.northwestern.edu',
    description='',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_myo_node = integrate_robot_human.robot_myo:main',
        ],
    },
)
