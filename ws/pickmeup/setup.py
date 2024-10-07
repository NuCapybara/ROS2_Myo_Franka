from setuptools import find_packages, setup

package_name = 'pickmeup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml',
                                    'launch/open_franka.launch.xml',
                                    'launch/grab_object.launch.xml',
                                    ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jialuyu',
    maintainer_email='=jialuyu2024@u.northwestern.edu',
    description='This Package is used to control the robot to pick and place objects',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['pick_place = pickmeup.pick_place:main',
                            'run = pickmeup.run:run_entry',
                            'testrun = pickmeup.testrun:main',
                            'delay_node = pickmeup.delay_node:delay_entry',

        ],
    },
)