"""Run preparation stages with periodic progress output."""
import subprocess
import time


def run_stage(command, label, **kwargs):
    started = time.monotonic()
    print(f'[{label}] 开始', flush=True)
    with subprocess.Popen(command, **kwargs) as proc:
        while True:
            try:
                code = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'[{label}] 进行中，已用 {time.monotonic() - started:.0f} 秒', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)
    print(f'[{label}] 完成，用时 {time.monotonic() - started:.0f} 秒', flush=True)
