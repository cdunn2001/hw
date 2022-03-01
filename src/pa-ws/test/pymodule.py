import os, re, subprocess


if not 'MODULEPATH' in os.environ:
	f = open(os.environ['MODULESHOME'] + "/init/.modulespath", "r")
	path = []
	for line in f.readlines():
		line = re.sub("#.*$", '', line)
		if line is not '':
			path.append(line)
	os.environ['MODULEPATH'] = ':'.join(path)

if not 'LOADEDMODULES' in os.environ:
	os.environ['LOADEDMODULES'] = ''
	
def module(*args):
	if type(args[0]) == type([]):
		args = args[0]
	else:
		args = list(args)
	(output, error) = subprocess.Popen(['/usr/bin/modulecmd', 'python'] + 
			args, stdout=subprocess.PIPE).communicate()
	exec (output)

