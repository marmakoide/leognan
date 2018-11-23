#! /usr/bin/env python

# the following two variables are used by the target "waf dist"
VERSION = '1.0.0'
APPNAME = 'leognan'

# these variables are mandatory ('/' are converted automatically)
top, out = '.', 'build'



def options(context):
	context.load('compiler_cxx')



def configure(context):
	context.load('compiler_cxx')
	context.env.CXXFLAGS = ['-std=c++14', '-Wall', '-Wextra', '-O3', '-g', '-frounding-math']
	context.check_cfg(package = 'eigen3', uselib_store = 'eigen', args = ['--cflags'])
	context.check_cxx(lib = 'm', cflags = '-Wall', uselib_store = 'm')



def build(context):
	# leognan library
	context(
		name            = 'leognan',
		includes        = 'include',
		export_includes = 'include'
	)

	# benchmarking program
	context.program(
		target   = 'leognan-benchmark',
		includes = 'benckmark',
		source   = context.path.ant_glob('benchmark/*.cpp'),
		use      = ['leognan', 'eigen', 'm']
	)

