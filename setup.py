from setuptools import find_packages, setup

package_name = "robotic_car"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ted",
    maintainer_email="ted.jh@hotmail.com",
    description="TODO: Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "controller = robotic_car.controller:main",
            "car = robotic_car.car:main",
            "data_collection_node = robotic_car.data_collection:main",
        ],
    },
)
