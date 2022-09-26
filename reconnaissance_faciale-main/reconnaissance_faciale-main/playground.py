required_libraries = {'os', 'cv2', 'pickle', 'copy'}

import sys
import subprocess
import pkg_resources

installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required_libraries - installed

if missing:
    #python = sys.executable
    #subprocess.check_call([python, 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    print(missing)
else:
    print("All the needed packages are installed")