import socket
import subprocess
from multiprocessing import Process

from .io import *
from .str import horizontal_line
from .time import *


def get_hostname():
    return socket.gethostname()


def get_hostaddr():
    return socket.gethostbyname(get_hostname())


def get_python_path():
    return subprocess.run("which python".split(), stdout=subprocess.PIPE).stdout.decode('utf-8').strip()


def run_command(command, title=None, c='-', mb=0, real=True):
    with MyTimer(name=f"run_command(title={title}, command={command})" if title else f"run_command(command={command})", t=0, b=0):
        print(horizontal_line(c=c))
        if real:
            os.system(command)
        print(horizontal_line(c=c, b=mb))


class GpuMemoryMonitor:
    def __init__(self, outfile, hostnames=(get_hostname(),), interval=1.0):
        self.outfile: Path = make_parent_dir(outfile)
        self.outfile.open('w').close()
        self.hostnames = hostnames
        self.interval = interval

    def log_gpu_usage(self):
        record = dict()
        for hostname in self.hostnames:
            cmd = f"ssh {hostname} nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            try:
                numbers = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT).decode('ascii').split('\n')[:-1]
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"command {e.cmd} return with error (code {e.returncode}): {e.output}")
            record[f'time@{hostname}'] = now(fmt='%Y-%m-%d %H:%M:%S')
            record.update({f'{hostname}:{i}': int(x) for i, x in enumerate(numbers)})
        with self.outfile.open('a') as out:
            out.write(json.dumps(record) + '\n')

    def monitor_gpu_usages(self):
        MyTimer(self.interval, self.monitor_gpu_usages).start()
        self.log_gpu_usage()


class AutoProcess(Process):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()
