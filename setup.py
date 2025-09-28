from setuptools import setup, find_packages

setup(
    name='zeit4150envs',
    version='0.1.0',
    author='Mo Hossny',
    author_email='drmohossny@gmail.com',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cmoxiv/ai-playground',
    packages=find_packages(),
    include_package_data=True,  # Required to include package_data
    package_data={
        'zeit4150envs': ['resources/*'],  # include all files in resources folder
    },
    scripts=['zeit4150-demo-random-agent-on-DOTA-maze.py',
             'zeit4150-demo-rl-agent-on-DOTA-maze.py'],
    install_requires=[
        'numpy>=1.21.0',
        'pygame>=2.0.0',
        'gymnasium>=0.29.1',
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
