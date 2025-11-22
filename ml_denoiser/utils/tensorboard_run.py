import sys
import subprocess
import time
from pathlib import Path


LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else Path.cwd() / "tensorboard_logs"
PORT = 6006

def launch_tensorboard(log_dir: Path, port: int) -> None:
    """Launch Tensorboard server pointing to the specified log directory."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist. Skipping Tensorboard launch.")
        return

    print(f"Launching Tensorboard server at port {port}, log dir: {log_dir}")

    tb_command = [
        "tensorboard",
        f"--logdir={str(log_dir)}",
        f"--port={port}",
        "--host=0.0.0.0",
    ]

    process = subprocess.Popen(tb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while True:
            if process.poll() is not None:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        print("Tensorboard server terminated.")

if __name__ == "__main__":
    launch_tensorboard(LOG_DIR, PORT)