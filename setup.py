from setuptools import setup

setup(
	name='pretrain',
	author='Russ Masuda',
	author_email='masuda3@hawaii.edu',
	packages=['pretrain'],
	install_requires=[
		'opencv-python',
		'numpy',
		'tensorflow',
		'keras',
		'matplotlib',
	],
)