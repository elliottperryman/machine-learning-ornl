import numpy as np
from os import system as sys
from time import sleep

# get the simWF id
id = ''
with open('simWF_ID', 'r') as file:
	id = file.read().split()
	id = id[0]

sleep(100)	
sys('./gdrive-linux-x64 list --no-header -q \"\''+str(id)+'\' in parents\" > dir_list')
sleep(100)	

# get list of directories
dirs = ''
with open('dir_list', 'r') as file:
	dirs = file.read().split()
	dirs = np.array(dirs)	
	dir_len = int(len(dirs)/5)	
	dirs = dirs.reshape(dir_len,5)		
	dirs = dirs[:,0]
	# get list of files
	for dir_id in dirs:
		sleep(100)	
		sys('./gdrive-linux-x64 list --no-header -q \"\''+str(dir_id)+'\' in parents\" > tmp_file_list')
		files = ''
		with open('tmp_file_list', 'r') as file:
			files = file.read().split()
			files = np.array(files)	
			file_len = int(len(files)/7)	
			try:	
				files = files.reshape(file_len,7)
				files = files[:,0]
			except:
				print("could not read dir: "+str(dir_id))
				continue

		for file_id in files:
		# manipulate file here	
			print(file_id)	

print("could not read directories:")
print(error_dirs)


