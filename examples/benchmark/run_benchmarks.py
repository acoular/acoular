import glob
import types
import subprocess
import socket
from os import path, mkdir

testmodules = glob.glob("test*.py")
machine = socket.gethostname()
if not path.exists("results"):
    mkdir("results")
if not path.exists(path.join("results", machine)):
    mkdir(path.join("results", machine))


def filt_func(f):
    return f.startswith("test") and isinstance(eval(f), types.FunctionType)


for testmodule in testmodules:
    mname = testmodule.split(".")[0]
    module = __import__(mname, globals(), locals(), [], 0)

    testfunctions = tuple(
        filter(
            lambda f: (
                f.startswith("test")
                and isinstance(module.__getattribute__(f), types.FunctionType)
            ),
            dir(module),
        )
    )
    for fname in testfunctions:
        ofnam = path.join("results", machine, f"{fname[5:]}.json")
        if path.exists(ofnam):
            print(f"{ofnam} exists, skipping")
            continue
        print(f"running {fname}:")
        subprocess.run(
            (
                "pyperf",
                "timeit",
                "-s",
                f"import {mname}",
                f"{mname}.{fname}()",
                "--append",
                ofnam,
                "--values=5",
                "--processes=2",
            )
        )
