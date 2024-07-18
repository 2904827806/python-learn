# 导入进程包
import os
import multiprocessing


def test1():
    print('子进程', os.getpid())  # 子进程
    print('父进程', os.getppid())  # 父进程
    for i in range(2):
        print('test1')


def test2():
    print('子进程', os.getpid())  # 子进程
    print('父进程', os.getppid())  # 父进程
    for i in range(2):
        print('test2')


if __name__ == '__main__':
    print('主进程', os.getpid())  # 主进程
    # 创建子进程
    test_1 = multiprocessing.Process(target=test1)
    # 创建子进程
    test_2 = multiprocessing.Process(target=test2)
    # 启动进程
    test_1.start()
    test_2.start()
