import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

col_names = ['PID', 'PPID', 'time', 'memory', 'command_name', 'full_command_name']
col_num = len(col_names)

### ________________________________________________
###
### deals with this strange ps output (process tree)
### ________________________________________________

def process_fline(line):
    # parse these last two messed up cols
    if "\\_ " in line:
        str = line.split(' ', col_num - 1)
        key = str[0]
        tmp = str[col_num - 1].split("\\_ ", 2)
        for i, t in enumerate(tmp):
            tmp[i] = t.strip()
        str[col_num - 2] = tmp[1]
        str[col_num - 1] = tmp[2]
    else:
        str = line.split(' ', col_num - 1)
        key = str[0]

    # rewrite int time in secs
    tmp1 = str[2].split(':', 1)
    tmp1 = [int(t) for t in tmp1]
    time = tmp1[1] + tmp1[0] * 60
    str[2] = time

    # rewrite float mem in MB
    str[3] = float(str[3]) / 1024.0

    add = np.array([[item] for item in str[1:]])
    return key, add

### ________________________________________________________
###
### returns a dictionary of numpy arrays where keys are pids
### ________________________________________________________

def parse_mem_usage_output(fpath):
    data = {}
    with open(fpath, 'r') as f:
        for line in f.readlines():
            key, add = process_fline(line)
            if key in data:
                data[key] = np.hstack((data[key], add))
            else:
                data[key] = add
    return data


def plot_data(data):
    for key, value in data.items():
        tmp = value.transpose()
        df = pd.DataFrame(data=tmp, index=None, columns=col_names[1:])
        title = df['command_name'][0]
        df = df.drop(columns=['PPID','command_name', 'full_command_name'])
        df = df.astype(float)

        fig = plt.figure(key)
        ax = fig.add_subplot(111)
        p = ax.plot(df['time'], df['memory'])
        plt.title(title)
        plt.xlabel("time")
        plt.grid()
        plt.ylabel("Memory (in MB)")
        fig.savefig(title + '.png')
    return 0

# ! add path
path_to_test_mem_file = ""
data = parse_mem_usage_output(path_to_test_mem_file)
plot_data(data)