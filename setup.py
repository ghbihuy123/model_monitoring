import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_obs",
    version="0.0.1",
    author="Lê Huỳnh Đức",
    author_email="duclh21@tpb.com.vn",
    description="ML observation module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab-icp4d.tpb.vn/DucLH21/ml_observation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)