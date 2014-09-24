
import threading
import Queue as Q
import datetime

# Demo Response Time of Parallel Computing on an additive operation
# MIMD - http://en.wikipedia.org/wiki/Flynn%27s_taxonomy
# O(N), O(log N), O(1)

# global settings
latency_analytics = 1.0/2  # unit to represent I/O and local calculations
latency_combine = 1.0/2  # unit to represent additive calculation, or things even more complicated to combine
suspend_print = False
png_folder = "output"

# global variables internally to collect statistics of consumption
stats_analytics = 0
stats_combine = 0
print_q = Q.Queue()
print_last_time = datetime.datetime.now()


def analytics(t):
    sync_print("%s: Analytics at %d" % (now_iso(), t))
    sleep(latency_analytics)

    global stats_analytics
    stats_analytics += 1

    return {'t': [t], 'a': t}


def combine(a1, a2):
    if a1 is None:
        return a2
    sync_print("%s: Add %s with %s" % (now_iso(), a1['t'], a2['t']))
    a = dict()
    a['t'] = sorted(a1['t'] + a2['t'])
    a['a'] = a1['a'] + a2['a']
    sleep(latency_combine)

    global stats_combine
    stats_combine += 1

    return a


def split(ts):  # cut T in the middle into two date intervals. Approximately,
    m = int(len(ts)/2)
    assert m > 0
    ts1 = ts[0:m]
    ts2 = ts[m:]
    return [ts1, ts2]


def divisible(ts):
    if len(ts) > 1:  # can be split into smaller intervals
        return True
    else:
        return False


# wrapper with a queue for threaded mode
def future_fn_qu(fn, ts, q):
    a = fn(ts)
    q.put(a)


def future_map(fn, ts2):
    a = []
    threads = []
    q = Q.Queue()
    for ts in ts2:
        th = threading.Thread(target=future_fn_qu, args=(fn, ts, q,))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    while not q.empty():
        a.append(q.get())
    assert len(a) == len(ts2)
    return a


def calc_analytics_p(ts):  # Future is distributed
    if divisible(ts):
        ts2 = split(ts)
        assert len(ts2) == 2
        sync_print("%s: Par2 %s %s" % (now_iso(), ts2[0], ts2[1]))

        method = "thread"
        if method == "thread":
            a = future_map(calc_analytics_p, ts2)
        else:
            a = map(calc_analytics_p, ts2)  # non-parallel default
        return combine(*a)
    else:
        sync_print("%s: Par1 %s" % (now_iso(), ts))
        return analytics(ts[0])  # e.g. one month or one day


# --------------------------------------------------
# ------------------------ O(N) --------------------
# --------------------------------------------------
def calc_linear(months):
    a = None
    for t in date_range(months):
        a = combine(a, analytics(t))
    return a


# --------------------------------------------------
# ------------------- O(log2(N)) -------------------
# --------------------------------------------------
def calc_parallel(months):
    ts = date_range(months)
    return calc_analytics_p(ts)


def generate_statistics(fn, iterations, min_months, max_months):
    global stats_analytics
    global stats_combine
    global suspend_print

    suspend_print = True
    results = []
    for i in range(iterations):
        import random
        from math import log, pow
        r1, r2 = [log(min_months, 2), log(max_months+1, 2)]  # max is exclusive
        m = int(pow(2, random.uniform(r1, r2)))
        time1 = now()

        stats_analytics = 0
        stats_combine = 0
        a = fn(m)
        time2 = now()

        delta = time2 - time1
        delta = delta.seconds + delta.microseconds/1E6
        results.append([m, delta, a])
        print "[%d] Elapsed time: %.2f seconds for %3d months" % (i, delta, m)
        if i % 10 == 0:
            plot_result(results, fn)
    print "Statistics finished"
    # raw_input("Press Enter to continue...")


def plot_result(results, fn):
    import numpy as np
    import matplotlib.pyplot as plt

    x1 = np.array([])
    y1 = np.array([])
    x2 = np.array([])
    y2 = np.array([])
    for rs in results:
        x1 = np.append(x1, rs[0])
        y1 = np.append(y1, rs[1])
        x2 = np.append(x2, np.log2(rs[0]))
        y2 = np.append(y2, rs[1])

    print "    Results len = %d, %d" % (np.alen(x1), np.alen(x2))

    plt.ioff()
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'yo')
    plt.title('Cloud Computing - O(T) vs O(log T) vs O(1)')
    plt.xlabel('Months')
    plt.ylabel('Time (s)')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'r.')
    plt.xlabel('Log2 Months')
    plt.ylabel('Time (s)')
    plt.draw()
    plt.ion()
    plt.show()

    from os import path
    global png_folder
    png = path.join(png_folder, fn.__name__ + '_log_behavior.png')
    plt.savefig(png)
    print "    Saved to %s" % png


def date_range(months):
    # this is a fake date range function
    return range(months)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
def usage():
    print """
    -l, --linear: linear mode
    -p, --parallel: parallel mode
    -s, --stats: statistical mode, recursively between 4 and 256 for 100 times and plot
    -h, --help: this help
    """


def main(argv):
    global stats_analytics
    global stats_combine

    import getopt
    try:
        opts, args = getopt.getopt(argv, "lpsh",
                                   ["linear", "parallel", "stats", "help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    if len(opts) == 0:
        print "ERROR: No option provided"
        usage()
        sys.exit(2)

    fn = None
    stats_mode = False
    for opt, arg in opts:
        if opt in ('-l', '--linear'):
            fn = calc_linear
        elif opt in ('-p', '--parallel'):
            fn = calc_parallel
        elif opt in ('-s', '--stats'):
            stats_mode = True
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        else:
            print "ERROR: Unknown option: %s" % opt
            usage()
            sys.exit(2)

    if stats_mode:
        generate_statistics(fn, 1000, 4, 256)
        sys.exit(0)
    else:
        sync_print("%s: Start" % now_iso())
        time1 = now()

        stats_analytics = 0
        stats_combine = 0
        months = 2**5
        a = fn(months)

        sync_print("%s: Done" % now_iso(), flush=True)
        time2 = now()

        delta = time2 - time1
        delta = delta.seconds + delta.microseconds/1E6
        print "Result: %s" % a
        print "Elapsed time: %.2f seconds" % delta
        print "Stats: %d analytics; %d combines" % (stats_analytics, stats_combine)

    sys.exit(0)

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


def sleep(n):
    import time
    time.sleep(n)


def now_iso():
    return now().isoformat(' ')


def now():
    return datetime.datetime.now()


def sync_print(s, flush=False):
    global suspend_print
    flush = True  # useless for now
    if suspend_print:
        return None
    else:
        print s


def sync_print2(s, flush=False):  # too much overhead, waste a lot of time
    global print_q, print_last_time
    tm = now()
    delta = tm - print_last_time
    delta = delta.seconds + delta.microseconds/1E6
    if delta > 1.5:
        flush = True

    if flush:
        lock = threading.Lock()
        lock.acquire()
        while not print_q.empty():
            print print_q.get()
        print_last_time = now()
        lock.release()
    else:
        print_q.put(s)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])