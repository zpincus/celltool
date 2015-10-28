import numpy

def ks_stat(data1, data2):
    data1, data2 = map(numpy.asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = numpy.sort(data1)
    data2 = numpy.sort(data2)
    data_all = numpy.concatenate([data1,data2])
    cdf1 = numpy.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (numpy.searchsorted(data2,data_all,side='right'))/(1.0*n2)
    d = numpy.max(numpy.absolute(cdf1-cdf2))
    return d

def bootstrap_ks_1_pop(pop, n):
    l = len(pop) / 2
    ri = numpy.random.randint
    stats_out = [ks_stat(pop[ri(l, size=l)], pop[ri(l, size=l)]) for i in range(n)]
    stats_out.sort()
    return stats_out

def bootstrap_ks_n_pops(pops, n):
    ri = numpy.random.randint
    stats_out = []
    l = len(pops)
    n_each = n/(l*(l-1)/2)
    for i, j in numpy.ndindex((l,l)):
        if i >= j: continue
        p1, p2 = pops[i], pops[j]
        l1, l2 = len(p1), len(p2)
        stats_out += [ks_stat(p1[ri(l1, size=l1)], p2[ri(l2, size=l2)]) for i in range(n_each)]
    stats_out.sort()
    return stats_out

def bootstrap_onetail_pval(stat, dist):
    index = numpy.searchsorted(dist, stat)
    return float(len(dist) - index) / len(dist)

def symmetric_comparison(pops, n=100000):
    ref_ks_vals = []
    for pop in pops:
        ref_ks_vals.append(bootstrap_ks_1_pop(pop, n))
    l = len(pops)
    pvals = numpy.zeros((l, l))
    for i, j in numpy.ndindex((l,l)):
        if i >= j: continue
        p1, p2 = pops[i], pops[j]
        r1, r2 = ref_ks_vals[i], ref_ks_vals[j]
        r = r1+r2
        r.sort()
        ks = ks_stat(p1, p2)
        p = bootstrap_onetail_pval(ks, r)
        pvals[i,j] = pvals[j,i] = p 
    return pvals

def compare_to_ref(pops, refs, n=100000):
    if len(refs) > 1:
        r = bootstrap_ks_n_pops(refs, n)
    else:
        r = bootstrap_ks_1_pop(refs[0], n)
    all_refs = numpy.concatenate(refs)
    pvals = []
    for pop in pops:
        ks = ks_stat(pop, all_refs)
        p = bootstrap_onetail_pval(ks, r)
        pvals.append(p)
    return pvals

