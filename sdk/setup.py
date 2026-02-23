from setuptools import setup, find_packages

setup(
    name="agentpay",
    version="0.1.0",
    author="AgentPay",
    description="Python SDK for AgentPay - AI Agent Payment Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://leofundmybot.dev",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.24",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
