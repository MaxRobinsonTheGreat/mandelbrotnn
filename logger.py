import os, sys
from datetime import datetime
from shutil import copyfile
from stat import S_IREAD, S_IRGRP, S_IROTH

class Logger:
    def __init__(self, file, dir=None):
        if dir is None:
            self.dir = './results/'+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(self.dir, exist_ok=True)
        else:
            self.dir = './'+dir

        self.copyFile(file)
        
        self.console = sys.stdout
        self.results = open(self.dir+"/results.txt", 'a+')
        sys.stdout = self
        print('LOGGING RUN RESULTS')

    def copyFile(self, file):
        copy_file = os.path.join(self.dir, os.path.basename(file))
        copyfile(file, copy_file)
        os.chmod(copy_file, S_IREAD|S_IRGRP|S_IROTH)

    def createDir(self, dir_name):
        os.makedirs(os.path.join(self.dir, dir_name), exist_ok=True)
   
    def write(self, message):
        self.console.write(message)
        self.results.write(message)

    def flush(self):
        self.console.flush()

    def __del__(self):
        self.results.close()
