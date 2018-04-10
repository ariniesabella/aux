import time
import os
from os.path import join, dirname, abspath
import subprocess


def which(program):
    """
    returns the path to an executable or None if it can't be found
    """
    def is_exe(_fpath):
        return os.path.isfile(_fpath) and os.access(_fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def main():
    output_fpath = join(dirname(abspath(__file__)), 'r_output.txt')
    rscript = which('Rscript')
    if rscript is None:
        print('No Rscript found\n')
        return 1

    pca_r = join(dirname(abspath(__file__)), 'test.R')

    cmdl = '{rscript} {pca_r} {output_fpath}'.format(**locals())
    subprocess.call([rscript, pca_r, output_fpath])

    start_time = cur_time = time.time()
    while cur_time - start_time < 15:
        with open("test.out", 'w') as f:
            f.write("test")
        cur_time = time.time()

    return 0


if __name__ == "__main__":
    main()