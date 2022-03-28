from datetime import *
from sys import stdout, stderr
from time import *
from typing import Optional


def elasped_sec(x, *args, **kwargs):
    t1 = datetime.now()
    return x(*args, **kwargs), datetime.now() - t1


def now(fmt='[%Y/%m/%d %H:%M:%S]'):
    return datetime.now().strftime(fmt)


def str_delta(x: timedelta):
    mm, ss = divmod(x.total_seconds(), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02.0f}:{mm:02.0f}:{ss:06.3f}"


class MyTimer:
    def __init__(self, prefix=None, name=None, t=0, b=1, logging=True):
        self.prefix = prefix
        self.name = name
        self.logging: bool = logging
        self.t: int = t
        self.b: int = b
        self.t1: Optional[datetime] = None
        self.t2: Optional[datetime] = None
        self.delta: Optional[timedelta] = None

    def __enter__(self):
        stdout.flush()
        stderr.flush()
        sleep(0.5)
        if self.name and self.logging:
            if self.t > 0:
                for _ in range(self.t):
                    print()
            print(f'{now()} {self.prefix + chr(32) if self.prefix else ""}[INIT] {self.name}')
        self.t1 = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t2 = datetime.now()
        self.delta = self.t2 - self.t1
        stdout.flush()
        stderr.flush()
        sleep(0.5)
        if self.name and self.logging:
            print(f'{now()} {self.prefix + chr(32) if self.prefix else ""}[EXIT] {self.name} ($={str_delta(self.delta)})')
            if self.b > 0:
                for _ in range(self.b):
                    print()
