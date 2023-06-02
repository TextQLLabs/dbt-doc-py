from setuptools import setup, find_packages

setup(
    name="dbt-doc-py",
    version="0.1.20",
    packages=find_packages(),
    install_requires=[
    "PyYAML", "dataclasses", "transformers", "argparse", "httpx", "inquirer", "ruamel.yaml",
    ],
    entry_points={
        "console_scripts": [
            "dbt-doc-py=dbt_doc_py.dbt_doc_py:run_async_main",
        ],
    },
    author="TextQLLabs",
    author_email="lenin@textql.com",
    description="Automatically generate docs for undocumented DBT models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TextQLLabs/dbt-doc-py.git",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",        
        "Programming Language :: Python :: 3.11",
    ],
)