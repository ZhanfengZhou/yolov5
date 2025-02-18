from setuptools import setup

package_name = 'yolo_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhanfeng',
    maintainer_email='zhanfeng.zhou@mail.utoronto.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolopub_xyz = yolo_detect.yolo_publisher_xyz:main",
            "yolocli_xyz = yolo_detect.yolo_client_xyz:main",
            "grasp_interface_yolo = yolo_detect.grasp_interface_yolo:main",
        ],
    },
)
