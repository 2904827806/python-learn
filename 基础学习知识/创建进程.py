#使用multiprocessing创建进程
from multiprocessing import Process
import os
import time

#使用multiprocessing创建进程
"""def stare(interval):
    t_time = time.time()
    time.sleep(interval)
    print('我是子进程')
    e_time = time.time()
    print('执行时间%s' % (e_time-t_time))


def main():
    print('主进程开始')
    p = Process(target=stare,args=(1,))
    p.start()
    print('主进程结束')
if __name__ == '__main__':
    main()"""
#使用multiprocessing创建两个进程
"""def chile_1(interval):
    print('子进程%s 开始执行，父进程为 %s' % (os.getpid(),os.getppid()))
    t = time.time()
    time.sleep(interval)
    e = time.time()
    print('子进程%s 执行时间为 %s ' % (os.getpid(),(e-t)))


def chile_2(interval):
    print('子进程%s 开始执行，父进程为 %s ' % (os.getpid(), os.getppid()))
    t = time.time()
    time.sleep(interval)
    e = time.time()
    print('子进程%s 执行时间为 %s ' % (os.getpid(), (e - t)))


def main():
    print('父进程（%s)的ID为（%s)' % ('FQ',os.getppid()))
    p1 = Process(target=chile_1,name='morsft',args=(1,))
    p2 = Process(target=chile_2, args=(1,))
    p1.start()
    p2.start()
    if p1.is_alive():
        print('p1 %s' % ('正在执行'))
        print('p1的ID为 %s' % p1.pid)
        print('p1的名称为 %s' % p1.name)
    if p2.is_alive():
        print('p2 %s' % ('正在执行'))
        print('p2的ID为 %s' % p2.pid)
        print('p2的名称为 %s' % p2.name)
    p1.join()
    p2.join()
    print('进程完成')"""

#使用Procese 的子类创建进程

"""class SubProcess(Process):
    def __init__(self,interval,name=''):
        print('我是进程')
        super(Process, self).__init__()
        self.interval = interval
        if name:
            self.name = name

    def run(self):
        print('子进程（%s)开始执行，父进程是（%s)' % (os.getpid(),os.getppid()))
        t = time.time()
        time.sleep(self.interval)
        e = time.time()
        print('子进程（%s) 执行时间是（%s)' % (os.getpid(),(e-t)))


def main():
    print('父进程开始执行，ID为: %s' % os.getppid())
    p1 = SubProcess(interval=1,name='mrsoft')
    p2 = SubProcess(interval=2)
    p1.start()
    p2.start()
    if p1.is_alive():
        print('p1 %s' % ('正在执行'))
        print('p1的ID为 %s' % p1.pid)
        print('p1的名称为 %s' % p1.name)
    if p2.is_alive():
        print('p2 %s' % ('正在执行'))
        print('p2的ID为 %s' % p2.pid)
        print('p2的名称为 %s' % p2.name)
    p1.join()
    p2.join()
    print('进程完成')


if __name__ == '__main__':
    main()"""


#使用进程池Pool 创建进程
"""from multiprocessing import Pool


def task(interval):
    print('子进程 %s 执行 task %s' % (os.getpid(),interval))
    time.sleep(1)


def main():
    print('执行父进程 %s' % os.getppid())
    p = Pool(3)
    for i in range(10):
        p.apply_async(task,args=(i,))

    print('进程结束')
    p.close()
    p.join()
    print('所有子进程结束')


if __name__ == '__main__':
    main()"""

#队列在进程之间的通信
from multiprocessing import Process, Queue


def write_task(q):
    if not q.full():
        for i in range(5):
            q.put("消息{}".format(i))
            print('写入消息{}'.format(i))
        print(q.qsize())
def read_task(q):
    time.sleep(1)
    while not q.empty():
        print('读取 %s' % q.get(True,2))
        print(q.qsize())


if __name__ == '__main__':
    print('父进程开始')
    q = Queue()
    pw = Process(target=write_task,args=(q,))
    pr = Process(target=read_task, args=(q,))
    pw.start()
    pr.start()
    pw.join()
    pr.join()
    print("主进程结束")

