"""Wait for the shared GPU lock and recheck free memory while holding it."""
import fcntl
import subprocess
import time


def acquire_gpu_lock(path, notify, minimum_free_mib=14500, poll_seconds=30):
    lock = path.open('a')
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                notify(state='waiting_for_gpu_lock', message='Shared GPU lock occupied; waiting, not failing')
                time.sleep(poll_seconds)
                continue
            try:
                free = int(subprocess.check_output(
                    ['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                     '--format=csv,noheader,nounits'], text=True).strip())
            except Exception as exc:
                fcntl.flock(lock, fcntl.LOCK_UN)
                notify(state='gpu_query_failed', message=str(exc))
                raise
            if free >= minimum_free_mib:
                notify(state='gpu_lock_acquired', free_mib=free,
                       message='Shared GPU lock acquired; free memory rechecked under lock')
                return lock
            fcntl.flock(lock, fcntl.LOCK_UN)
            notify(state='waiting_for_gpu_memory', free_mib=free,
                   message='Insufficient memory after acquiring lock; released lock and waiting')
            time.sleep(poll_seconds)
    except BaseException:
        lock.close()
        raise
