from setuptools import setup

setup(
	name='train',
	author='Russ Masuda',
	author_email='masuda3@hawaii.edu',
	packages=['train'],
	install_requires=[
		'opencv-python',
		'numpy',
		'tensorflow',
		'keras',
		'matplotlib',
	],
)