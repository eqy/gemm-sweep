import torch
import torch.utils.benchmark as benchmark

def run_gemm(x, y):
    return torch.mm(x, y)

def run_baddbmm(z, x, y):
    return torch.baddbmm(z, x, y, alpha=0.001, beta=0.1)

def bench_gemm(results):
    max_n = 8192
    max_k = 8192
    max_m = 8192
    n = 1
    while n <= max_n:
        k = 1
        while k <= max_n:
            m = 1
            while m <= max_k:
                label = 'bench_gemm'
                sub_label = f'[{n}, {k}, {m}]'
                x = torch.randn(n, k, device='cuda', dtype=torch.half)
                y = torch.randn(k, m, device='cuda', dtype=torch.half)
                results.append(benchmark.Timer(
                    stmt='run_gemm(x, y)',
                    setup='from __main__ import run_gemm',
                    globals={'x': x, 'y': y},
                    label=label,
                    sub_label=sub_label,
                    description='time').blocked_autorange(min_run_time=0.2))
                m *= 8
            k *= 8
        n *= 8

def bench_baddbmm(results):
    max_b = 512
    max_n = 4096
    max_k = 4096
    max_m = 4096
    max_numel = 1073741824
    b = 1
    while b <= max_b:
        n = 1
        while n <= max_n:
            k = 1
            while k <= max_n:
                m = 1
                while m <= max_k:
                    if (b*n*k > max_numel):
                        continue
                    elif (b*k*m > max_numel):
                        continue
                    elif (b*n*m > max_numel):
                        continue
                    label = 'bench_baddbmm'
                    sub_label = f'[{b}, {n}, {k}, {m}]'
                    z = torch.randn(b, n, m, device='cuda', dtype=torch.half)
                    x = torch.randn(b, n, k, device='cuda', dtype=torch.half)
                    y = torch.randn(b, k, m, device='cuda', dtype=torch.half)
                    results.append(benchmark.Timer(
                        stmt='run_baddbmm(z, x, y)',
                        setup='from __main__ import run_baddbmm',
                        globals={'z': z, 'x': x, 'y': y},
                        label=label,
                        sub_label=sub_label,
                        description='time').blocked_autorange(min_run_time=0.2))
                    m *= 8
                k *= 8
            n *= 8
        b *= 8

def main():
    results = list()
    bench_gemm(results)
    bench_baddbmm(results)
    compare = benchmark.Compare(results)
    compare.print()

if __name__ == '__main__':
    main()
