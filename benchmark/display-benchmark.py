import sys
import numpy

import matplotlib.pyplot as plot



def load_data(in_file):
	ret = { }
	name, value_list = None, []

	for line in in_file:
		line = line.strip()
		if len(line) > 0:
			if line[0] == '#':
				if name is not None:
					ret[name] = numpy.array(value_list)
				name, value_list = line.split()[1], []
			else:
				value_list.append(float(line))

	if name is not None:
		ret[name] = numpy.array(value_list)	

	return ret



def display_data(data):
	fig, axes = plot.subplots(nrows = len(data), ncols = 1)
	
	for i, (name, value_list) in enumerate(data.iteritems()):
		ax = axes[i]
		ax.semilogy(numpy.arange(value_list.shape[0]), value_list, lw = 1., c = 'k')
		ax.set_title(name)

	plot.show()


def main():
	display_data(load_data(sys.stdin))


if __name__ == '__main__':
	main()
