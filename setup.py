'''
The setup.py file is an essential part of packaging and distributing Python projects.
It is used by setup tools (or disutils in older python versions) to define the configuration of yor project, 
such as its metadata, dependencies, and more
)
'''

from setuptools import find_packages,setup   #Find packages will search for the folders which contains __init__.py if exist hem it'll treat that filder as packages
from typing import List

def get_requirements()->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirement_lst : List[str] = []

    try:
        with open('requirements.txt','r') as file:
            ## Read lines from the file
            lines = file.readlines()

            for line in lines:
                requirement = line.strip()
                ##ignore empty lines and -e .
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found')                

    return requirement_lst


setup(
    name='NetworkSecurity_ML_Project',
    version='0.0.1',
    author= 'Sudharshan Paul',
    author_email='gantasudarshanpaul@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()
)