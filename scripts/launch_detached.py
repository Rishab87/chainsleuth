"""Start a command in its own session so it survives the parent shell / agent exiting.

    uv run python scripts/launch_detached.py logs/out.log -- <command...>
"""

import subprocess
import sys

log, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
with open(log, "w") as fh:
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
print(proc.pid)
