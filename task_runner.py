import signal
signal.signal(signal.SIGINT, signal.default_int_handler)

import redis
from sherlock import RedisLock
import sys
import subprocess
from time import sleep
import pickle as pkl
from pathlib import Path

if __name__ == '__main__':
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=15, password='nopass')
    lock = RedisLock('runner_lock', timeout=3600, expire=36000, client=redis_client, retry_interval=0.2)
    cmds_path = Path(sys.argv[1])
    chosen_path = Path(sys.argv[2])
    done_path = Path(sys.argv[3])
    fail_path = Path(sys.argv[4])
    gpu_limit = sys.argv[5]
    # cpu_limit = sys.argv[6]
    
    while True:
        print('(pre) waiting for lock', flush=True)
        lock.acquire()
        sleep(0.5)
        with open(cmds_path, 'r') as f:
            cmds = f.read().strip().split('\n')
        if not chosen_path.exists():
            chosen = set()
        else:
            with open(chosen_path, 'rb') as f:
                chosen = pkl.load(f)
        if not done_path.exists():
            done = set()
        else:
            with open(done_path, 'rb') as f:
                done = pkl.load(f)
        if not fail_path.exists():
            failed = set()
        else:
            with open(fail_path, 'rb') as f:
                failed = pkl.load(f)
        not_cand = chosen | done | failed

        selected = None
        for i, cmd in enumerate(cmds):
            if cmd.strip != "" and cmd not in not_cand:
                selected = cmd
                print('{}: {}'.format(i, cmd))
                break

        if selected is None:
            print('no more jobs', flush=True)
            lock.release()
            print('(pre) released lock', flush=True)
            exit(0)
        
        chosen.add(selected)
        with open(chosen_path, 'wb') as f:
            pkl.dump(chosen, f)
        sleep(0.5)
        lock.release()
        print('(pre) released lock', flush=True)
        
        task_errored = False
        interrupted = False
        try:
            # subprocess.run(f'taskset -c {cpu_limit} {selected}{gpu_limit}', shell=True, check=True)
            subprocess.run(f'CUDA_VISIBLE_DEVICES={gpu_limit} {selected}', shell=True, check=True)
        except KeyboardInterrupt:
            task_errored = True
            interrupted = True
        except subprocess.CalledProcessError:
            task_errored = True

        print('(post) waiting for lock', flush=True)
        lock.acquire()
        sleep(0.5)
        
        with open(chosen_path, 'rb') as f:
            chosen = pkl.load(f)
        chosen.remove(selected)
        if task_errored:
            if not fail_path.exists():
                failed = set()
            else:
                with open(fail_path, 'rb') as f:
                    failed = pkl.load(f)
            failed.add(selected)
        else:
            if not done_path.exists():
                done = set()
            else:
                with open(done_path, 'rb') as f:
                    done = pkl.load(f)
            done.add(selected)
        with open(chosen_path, 'wb') as f:
            pkl.dump(chosen, f)
        if task_errored:
            with open(fail_path, 'wb') as f:
                pkl.dump(failed, f)
        else:
            with open(done_path, 'wb') as f:
                pkl.dump(done, f)

        sleep(0.5)
        lock.release()
        print('(post) released lock', flush=True)
        
        if interrupted:
            break
    