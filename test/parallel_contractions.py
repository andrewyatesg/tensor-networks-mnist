import numpy as np
import tensornetwork as tn
import sys
import time
import threading
from itertools import count


def foreach(f, l, threads=3, return_=False):
    """
    Apply f to each element of l, in parallel
    """

    if threads > 1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(), l.__iter__())
        else:
            i = l.__iter__()

        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = next(i)
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n, x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()

        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise a(b).with_traceback(c)
        if return_:
            r = d.items()
            return [v for (n, v) in sorted(r)]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return


if __name__ == "__main__":
    num_threads = 1

    iter_barrier = threading.Barrier(num_threads + 1)

    def thread_main(ithread):
        # nodeA = tn.Node(A)
        # nodeB = tn.Node(B)
        # nodeA[0] ^ nodeB[1]
        # tn.contractors.auto([nodeA, nodeB], ignore_edge_order=True)
        A = np.ones((10000, 10000))
        B = np.ones((10000, 10000))
        C = np.zeros_like(A)
        np.dot(A, B, out=C)
        iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]
    for t in worker_threads:
        t.start()
    iter_barrier.wait()

    for t in worker_threads:
        t.join()