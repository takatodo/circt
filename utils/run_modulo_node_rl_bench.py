#!/usr/bin/env python3
"""Generate and run resource-constrained Modulo NodeRL microbenchmarks.

This intentionally measures scheduler behavior without requiring a Calyx-to-RTL
flow. Each instance contains independent loop-carried chains and joins that
simultaneously request two resources. The latter is a deliberate boundary:
the current simplex scheduler only supports one limited resource per node,
whereas CP-SAT provides an exact modulo-scheduling reference for small cases.
"""

import argparse
import itertools
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def resource_name(base, group, resource_clusters):
  return base if resource_clusters == 1 else f'{base}{group}'


def build_polybench_problem(kernel, chains, unroll, mul_limit, mul_ii,
                            div_limit, div_ii, simplex_compatible,
                            resource_clusters):
  """Build an SSP proxy for a tiled PolyBench inner loop.

  `chains` denotes independent output tiles and `unroll` the scalar work kept
  in one modulo-scheduled iteration. The proxies preserve the dependency shape
  of the named kernels while deliberately omitting memory banking; it is a
  scheduler benchmark, not a functional PolyBench implementation.
  """
  lines = [
      f'ssp.instance @poly_{kernel.replace("-", "_")} of "ModuloProblem" {{',
      '  library {',
      '    operator_type @mul [latency<2>]',
      '    operator_type @add [latency<1>]',
      '    operator_type @div [latency<3>]',
      '    operator_type @sink [latency<1>]',
      '  }',
      '  resource {',
  ]
  for group in range(resource_clusters):
    mul_resource = resource_name('mul_r', group, resource_clusters)
    div_resource = resource_name('div_r', group, resource_clusters)
    lines.append(f'    resource_type @{mul_resource} [limit<{mul_limit}>, '
                 f'ii<{1 if simplex_compatible else mul_ii}>]')
    lines.append(f'    resource_type @{div_resource} [limit<{div_limit}>, '
                 f'ii<{1 if simplex_compatible else div_ii}>]')
  lines.extend(['  }', '  graph {'])
  results = []
  for tile in range(chains):
    group = tile % resource_clusters
    mul_resource = resource_name('mul_r', group, resource_clusters)
    second_resource = (mul_resource if simplex_compatible else resource_name(
        'div_r', group, resource_clusters))
    if kernel == 'gemm':
      # C[i,j] += A[i,k] * B[k,j]: a partially-unrolled K reduction.
      previous = f'@acc{tile}_{unroll - 1}'
      for k in range(unroll):
        mul = f'%mul{tile}_{k}'
        acc = f'%acc{tile}_{k}'
        lines.append(f'    {mul} = operation<@mul>() uses[@{mul_resource}]')
        source = (f'{mul}, {previous} [dist<1>]'
                  if k == 0 else f'%acc{tile}_{k - 1}, {mul}')
        lines.append(f'    {acc} = operation<@add> @acc{tile}_{k}('
                     f'{source}) uses[@{second_resource}]')
      results.append(f'%acc{tile}_{unroll - 1}')
    elif kernel == '2mm':
      # tmp[i,j] += A[i,k]*B[k,j]; D[i,j] += tmp[i,k]*C[k,j].
      first_previous = f'@tmp{tile}_{unroll - 1}'
      second_previous = f'@out{tile}_{unroll - 1}'
      for k in range(unroll):
        product = f'%ab{tile}_{k}'
        tmp = f'%tmp{tile}_{k}'
        projected = f'%tmpc{tile}_{k}'
        out = f'%out{tile}_{k}'
        lines.append(f'    {product} = operation<@mul>() uses[@{mul_resource}]')
        first_source = (f'{product}, {first_previous} [dist<1>]'
                        if k == 0 else f'%tmp{tile}_{k - 1}, {product}')
        lines.append(f'    {tmp} = operation<@add> @tmp{tile}_{k}('
                     f'{first_source}) uses[@{second_resource}]')
        lines.append(f'    {projected} = operation<@mul>({tmp}) '
                     f'uses[@{mul_resource}]')
        second_source = (f'{projected}, {second_previous} [dist<1>]'
                         if k == 0 else f'%out{tile}_{k - 1}, {projected}')
        lines.append(f'    {out} = operation<@add> @out{tile}_{k}('
                     f'{second_source}) uses[@{second_resource}]')
      results.append(f'%out{tile}_{unroll - 1}')
    elif kernel == 'jacobi-2d':
      # Four-point stencil followed by a value carried to the next timestep.
      previous = f'@state{tile}_{unroll - 1}'
      for k in range(unroll):
        north = f'%north{tile}_{k}'
        west = f'%west{tile}_{k}'
        pair = f'%pair{tile}_{k}'
        state = f'%state{tile}_{k}'
        lines.append(f'    {north} = operation<@mul>() '
                     f'uses[@{mul_resource}]')
        lines.append(f'    {west} = operation<@mul>() '
                     f'uses[@{mul_resource}]')
        lines.append(f'    {pair} = operation<@add>({north}, {west}) '
                     f'uses[@{second_resource}]')
        source = (f'{pair}, {previous} [dist<1>]'
                  if k == 0 else f'%state{tile}_{k - 1}, {pair}')
        lines.append(f'    {state} = operation<@add> @state{tile}_{k}('
                     f'{source}) uses[@{second_resource}]')
      results.append(f'%state{tile}_{unroll - 1}')
    else:  # covariance
      # Mean reduction, normalization, then centered outer-product update.
      previous = f'@mean{tile}_{unroll - 1}'
      for k in range(unroll):
        mean = f'%mean{tile}_{k}'
        norm = f'%norm{tile}_{k}'
        product = f'%cov{tile}_{k}'
        source = (f'{previous} [dist<1>]' if k == 0 else f'%mean{tile}_{k - 1}')
        lines.append(f'    {mean} = operation<@add> @mean{tile}_{k}('
                     f'{source}) uses[@{second_resource}]')
        lines.append(
            f'    {norm} = operation<@div>({mean}) uses[@{second_resource}]')
        lines.append(f'    {product} = operation<@mul>({norm}) '
                     f'uses[@{mul_resource}]')
      results.append(f'%cov{tile}_{unroll - 1}')
  lines.append('    operation<@sink> @last(' + ', '.join(results) + ')')
  lines.extend(['  }', '}'])
  return '\n'.join(lines) + '\n'


def build_problem(kernel, chains, chain_length, mul_limit, mul_ii, div_limit,
                  div_ii, simplex_compatible, resource_clusters):
  if kernel != 'synthetic':
    return build_polybench_problem(kernel, chains, chain_length, mul_limit,
                                   mul_ii, div_limit, div_ii,
                                   simplex_compatible, resource_clusters)
  lines = [
      'ssp.instance @generated of "ModuloProblem" {',
      '  library {',
      '    operator_type @mul [latency<2>]',
      '    operator_type @div [latency<3>]',
      '    operator_type @join [latency<1>]',
      '    operator_type @sink [latency<1>]',
      '  }',
      '  resource {',
  ]
  for group in range(resource_clusters):
    mul_resource = resource_name('mul_r', group, resource_clusters)
    div_resource = resource_name('div_r', group, resource_clusters)
    lines.append(f'    resource_type @{mul_resource} [limit<{mul_limit}>, '
                 f'ii<{1 if simplex_compatible else mul_ii}>]')
    lines.append(f'    resource_type @{div_resource} [limit<{div_limit}>, '
                 f'ii<{1 if simplex_compatible else div_ii}>]')
  lines.extend(['  }', '  graph {'])
  joins = []
  for chain in range(chains):
    group = chain % resource_clusters
    mul_resource = resource_name('mul_r', group, resource_clusters)
    div_resource = resource_name('div_r', group, resource_clusters)
    second_resource = mul_resource if simplex_compatible else div_resource
    previous_mul = f'%m{chain}_0'
    previous_div = f'%d{chain}_0'
    lines.append(f'    {previous_mul} = operation<@mul> @m{chain}_0('
                 f'@m{chain}_0 [dist<1>]) uses[@{mul_resource}]')
    lines.append(f'    {previous_div} = operation<@div> @d{chain}_0('
                 f'@d{chain}_0 [dist<1>]) uses[@{second_resource}]')
    for stage in range(1, chain_length):
      mul = f'%m{chain}_{stage}'
      div = f'%d{chain}_{stage}'
      lines.append(f'    {mul} = operation<@mul>({previous_mul}) '
                   f'uses[@{mul_resource}]')
      lines.append(f'    {div} = operation<@div>({previous_div}) '
                   f'uses[@{second_resource}]')
      previous_mul, previous_div = mul, div
    join = f'%j{chain}'
    joins.append(join)
    resources = (f'@{mul_resource}'
                 if simplex_compatible else f'@{mul_resource}, @{div_resource}')
    lines.append(f'    {join} = operation<@join>({previous_mul}, '
                 f'{previous_div}) uses[{resources}]')
  lines.append('    operation<@sink> @last(' + ', '.join(joins) + ')')
  lines.extend(['  }', '}'])
  return '\n'.join(lines) + '\n'


def run(circt_opt, mlir, scheduler, episodes, episode_node_budget,
        resource_ordering_budget, local_search_nodes, local_search_time_limit,
        seed, timeout):
  scheduler_options = 'last-op-name=last'
  if scheduler == 'node-rl':
    scheduler_options += (
        f',episodes={episodes},seed={seed},'
        f'episode-node-budget={episode_node_budget},'
        f'resource-ordering-budget={resource_ordering_budget},'
        f'local-search-nodes={local_search_nodes},'
        f'local-search-time-limit={local_search_time_limit},'
        'learning-rate=0.05,exploration=1.0')
  elif scheduler == 'cpsat':
    scheduler_options += (f',time-limit={timeout},report-statistics=true')
  command = [
      str(circt_opt),
      str(mlir),
      f'-ssp-schedule=scheduler={scheduler} options={scheduler_options}',
  ]
  process_timeout = timeout + 1.0 if scheduler == 'cpsat' else timeout
  started = time.perf_counter()
  try:
    result = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=process_timeout)
  except subprocess.TimeoutExpired:
    return ('timeout', time.perf_counter() - started, None, None, None, 0, '')
  elapsed = time.perf_counter() - started
  lower_bound = re.search(r'lower-bound=(\d+)', result.stderr)
  ii = re.search(r'\[II<(\d+)>\]', result.stdout)
  latency = re.search(r'operation<@sink> @last\([^\n]*\) \[t<(\d+)>\]',
                      result.stdout)
  bindings = result.stdout.count('bindings<')
  return (result.returncode,
          elapsed, int(lower_bound.group(1)) if lower_bound else None,
          int(ii.group(1)) if ii else None,
          int(latency.group(1)) if latency else None, bindings, result.stderr)


def is_dominated(point, points):
  cost, ii, latency = point
  return any(other != point and other[0] <= cost and other[1] <= ii and
             other[2] <= latency and other != point for other in points)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--circt-opt', type=Path, required=True)
  parser.add_argument(
      '--kernel',
      choices=['synthetic', 'gemm', '2mm', 'jacobi-2d', 'covariance'],
      default='synthetic')
  parser.add_argument('--chains', type=int, nargs='+', default=[2, 4, 8])
  parser.add_argument('--chain-length', type=int, default=8)
  parser.add_argument('--unrolls',
                      type=int,
                      nargs='+',
                      help='inner-loop partial-unroll factors (defaults to '
                      '--chain-length)')
  parser.add_argument('--episodes', type=int, nargs='+', default=[32, 256])
  parser.add_argument('--episode-node-budget', type=int, default=65536)
  parser.add_argument('--resource-ordering-budget', type=int, default=1048576)
  parser.add_argument('--local-search-nodes', type=int, default=0)
  parser.add_argument('--local-search-time-limit', type=float, default=0.1)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--seeds', type=int, nargs='+')
  parser.add_argument('--timeout', type=float, default=10.0)
  parser.add_argument('--cpsat',
                      action='store_true',
                      help='also run the exact CP-SAT modulo scheduler')
  parser.add_argument('--simplex-compatible', action='store_true')
  parser.add_argument('--mul-limit', type=int, default=2)
  parser.add_argument('--mul-limits', type=int, nargs='+')
  parser.add_argument('--mul-ii', type=int, default=3)
  parser.add_argument('--mul-cost', type=int, default=3)
  parser.add_argument('--div-limit', type=int, default=1)
  parser.add_argument('--div-limits', type=int, nargs='+')
  parser.add_argument('--div-ii', type=int, default=2)
  parser.add_argument('--div-cost', type=int, default=5)
  parser.add_argument(
      '--resource-clusters',
      type=int,
      default=1,
      help='split chains across this many disjoint resource-type groups')
  args = parser.parse_args()
  if args.resource_clusters <= 0:
    parser.error('--resource-clusters must be positive')
  if args.episode_node_budget < 0:
    parser.error('--episode-node-budget must be nonnegative')
  if args.resource_ordering_budget < 0:
    parser.error('--resource-ordering-budget must be nonnegative')
  if args.local_search_nodes < 0:
    parser.error('--local-search-nodes must be nonnegative')
  if args.local_search_time_limit <= 0.0:
    parser.error('--local-search-time-limit must be positive')
  if args.seed < 0 or (args.seeds and any(seed < 0 for seed in args.seeds)):
    parser.error('--seed/--seeds must be nonnegative')
  if any(chains < args.resource_clusters for chains in args.chains):
    parser.error('--resource-clusters cannot exceed --chains')
  print(
      'scheduler,kernel,mul_limit,div_limit,mul_ii,div_ii,cost,chains,unroll,nodes,resource_clusters,episodes,seed,episode_node_budget,resource_ordering_budget,local_search_nodes,local_search_time_limit,status,lower_bound,ii,latency,static_bindings,elapsed_seconds'
  )
  pareto = []
  measurements = {}
  with tempfile.TemporaryDirectory(prefix='circt-node-rl-') as directory:
    mul_limits = args.mul_limits or [args.mul_limit]
    div_limits = args.div_limits or [args.div_limit]
    unrolls = args.unrolls or [args.chain_length]
    for mul_limit in mul_limits:
      for div_limit in div_limits:
        for chains, unroll in itertools.product(args.chains, unrolls):
          mlir = Path(directory) / f'chains-{chains}-unroll-{unroll}.mlir'
          mlir.write_text(
              build_problem(args.kernel, chains, unroll, mul_limit, args.mul_ii,
                            div_limit, args.div_ii, args.simplex_compatible,
                            args.resource_clusters))
          if args.kernel == 'synthetic':
            nodes = chains * (2 * unroll + 1) + 1
          elif args.kernel == 'gemm':
            nodes = chains * 2 * unroll + 1
          elif args.kernel == '2mm':
            nodes = chains * 4 * unroll + 1
          elif args.kernel == 'jacobi-2d':
            nodes = chains * 4 * unroll + 1
          else:
            nodes = chains * 3 * unroll + 1
          cost = mul_limit * args.mul_cost
          if not args.simplex_compatible:
            cost += div_limit * args.div_cost
          cost *= args.resource_clusters
          effective_mul_ii = 1 if args.simplex_compatible else args.mul_ii
          effective_div_ii = 1 if args.simplex_compatible else args.div_ii
          schedulers = ['node-rl']
          if args.simplex_compatible:
            schedulers.append('simplex')
          if args.cpsat:
            schedulers.append('cpsat')
          for scheduler in schedulers:
            for episodes in args.episodes if scheduler == 'node-rl' else [0]:
              seeds = (args.seeds or [args.seed]
                       if scheduler == 'node-rl' else [0])
              for seed in seeds:
                code, elapsed, lower_bound, ii, latency, bindings, stderr = run(
                    args.circt_opt, mlir, scheduler, episodes,
                    args.episode_node_budget, args.resource_ordering_budget,
                    args.local_search_nodes, args.local_search_time_limit, seed,
                    args.timeout)
                if code == 'timeout':
                  status = 'timeout'
                else:
                  cp_status = re.search(r'cpsat: status=(\w+)', stderr)
                  if code == 0 and ii is not None:
                    status = (cp_status.group(1) if scheduler == 'cpsat' and
                              cp_status else 'feasible')
                  elif scheduler == 'cpsat' and 'time limit exceeded' in stderr:
                    status = 'unknown'
                  else:
                    status = 'failed'
                print(
                    f'{scheduler},{args.kernel},{mul_limit},{div_limit},'
                    f'{effective_mul_ii},{effective_div_ii},{cost},{chains},'
                    f'{unroll},{nodes},{args.resource_clusters},{episodes},'
                    f'{seed},{args.episode_node_budget},'
                    f'{args.resource_ordering_budget},'
                    f'{args.local_search_nodes},'
                    f'{args.local_search_time_limit},{status},'
                    f'{lower_bound or ""},{ii or ""},{latency or ""},'
                    f'{bindings},{elapsed:.6f}',
                    flush=True)
                summary_key = (scheduler, args.kernel, mul_limit, div_limit,
                               effective_mul_ii, effective_div_ii, cost, chains,
                               unroll, nodes, args.resource_clusters, episodes,
                               args.local_search_nodes,
                               args.local_search_time_limit)
                measurements.setdefault(summary_key, []).append(
                    (status, lower_bound, ii, latency, elapsed))
                if status in ('feasible', 'optimal'):
                  pareto.append(
                      (scheduler, args.kernel, cost, ii, latency, mul_limit,
                       div_limit, effective_mul_ii, effective_div_ii, chains,
                       unroll, episodes, args.local_search_nodes,
                       args.local_search_time_limit, args.resource_clusters,
                       seed))
                if status == 'failed':
                  diagnostic = next(
                      (line for line in stderr.splitlines()
                       if 'error:' in line),
                      stderr.splitlines()[0] if stderr else 'no diagnostic')
                  print(f'# diagnostic: {diagnostic}', flush=True)
  print(
      'summary,scheduler,kernel,mul_limit,div_limit,mul_ii,div_ii,cost,chains,unroll,nodes,resource_clusters,episodes,local_search_nodes,local_search_time_limit,runs,success_rate,lower_bound,min_ii,median_ii,max_ii,min_latency,median_latency,max_latency,mean_elapsed_seconds'
  )
  for key, results in sorted(measurements.items()):
    successful = [
        result for result in results if result[0] in ('feasible', 'optimal')
    ]
    lower_bounds = [result[1] for result in successful if result[1] is not None]
    iis = [result[2] for result in successful]
    latencies = [result[3] for result in successful]

    def metric(values, function):
      return '' if not values else f'{function(values):g}'

    print('summary,' + ','.join(map(str, key)) +
          f',{len(results)},{len(successful) / len(results):.6f},'
          f'{metric(lower_bounds, min)},{metric(iis, min)},'
          f'{metric(iis, statistics.median)},'
          f'{metric(iis, max)},{metric(latencies, min)},'
          f'{metric(latencies, statistics.median)},'
          f'{metric(latencies, max)},'
          f'{statistics.mean(result[4] for result in results):.6f}')
  if pareto:
    print(
        'pareto,scheduler,kernel,cost,ii,latency,mul_limit,div_limit,mul_ii,div_ii,chains,unroll,episodes,local_search_nodes,local_search_time_limit,resource_clusters,seed'
    )
    workloads = sorted(
        set((point[0], point[1], point[9], point[10], point[14], point[15])
            for point in pareto))
    for scheduler, kernel, chains, unroll, resource_clusters, seed in workloads:
      points = [
          point for point in pareto
          if (point[0], point[1], point[9], point[10], point[14],
              point[15]) == (scheduler, kernel, chains, unroll,
                             resource_clusters, seed)
      ]
      for point in points:
        if not is_dominated(point[2:5], [other[2:5] for other in points]):
          print('pareto,' + ','.join(map(str, point)))


if __name__ == '__main__':
  main()
