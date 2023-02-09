# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing
import os
import time


def work(name):
    print('子进程work正在运行......')
    print(multiprocessing.current_process().name.split('-'))
    time.sleep(0.5)
    print(name)
    # 获取进程的名称
    print('子进程name', multiprocessing.current_process())
    # 获取进程的pid
    print('子进程pid', multiprocessing.current_process().pid, os.getpid())
    # 获取父进程的pid
    print('父进程pid', os.getppid())
    print('子进程运行结束......')


if __name__ == '__main__':
    print('主进程启动')
    # 获取进程的名称
    print('主进程name', multiprocessing.current_process())
    # 获取进程的pid
    print('主进程pid', multiprocessing.current_process().pid, os.getpid())
    # 创建进程
    p = multiprocessing.Process(group=None, target=work, args=('tigeriaf', ))
    # 启动进程
    p.start()
    print('主进程结束')
