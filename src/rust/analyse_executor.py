import re
import sys
import statistics
from collections import defaultdict

BATCH_RE = re.compile(
    r"\[TRACE\]\s+t=([\d.]+)s\s+exec=(\d+)\s+"
    r"event=BATCH_START\s+batch=(\d+)/(\d+)"
)

EVAL_RE = re.compile(
    r"\[TRACE\]\s+t=([\d.]+)s\s+exec=(\d+)\s+"
    r"event=EVAL_DONE\s+batch=(\d+)/(\d+)\s+eval_ms=([\d.]+)"
)

OLD_BATCH_RE = re.compile(r"\[Executor\s+(\d+)\]\s+batch_size=(\d+)/(\d+)")


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values))

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def stats(values):
    if not values:
        return None

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def fmt(s):
    if s is None:
        return "n/a"

    return (
        f"n={s['n']} "
        f"mean={s['mean']:.3f} "
        f"median={s['median']:.3f} "
        f"p10={s['p10']:.3f} "
        f"p90={s['p90']:.3f} "
        f"min={s['min']:.3f} "
        f"max={s['max']:.3f}"
    )


text = sys.stdin.read()

batches = []
evals = []

for m in BATCH_RE.finditer(text):
    batches.append(
        {
            "t": float(m.group(1)),
            "exec": int(m.group(2)),
            "size": int(m.group(3)),
            "max": int(m.group(4)),
        }
    )

for m in EVAL_RE.finditer(text):
    evals.append(
        {
            "t": float(m.group(1)),
            "exec": int(m.group(2)),
            "size": int(m.group(3)),
            "max": int(m.group(4)),
            "eval_ms": float(m.group(5)),
        }
    )


# Fall back to old output format if no TRACE batches were found.
if not batches:
    for m in OLD_BATCH_RE.finditer(text):
        batches.append(
            {
                "t": None,
                "exec": int(m.group(1)),
                "size": int(m.group(2)),
                "max": int(m.group(3)),
            }
        )


if not batches:
    print("No executor batch lines found.")
    sys.exit(1)


max_batch = batches[0]["max"]

sizes = [x["size"] for x in batches]

print("=" * 72)
print("EXECUTOR SUMMARY")
print("=" * 72)

print(f"Total batches:       {len(batches):,}")
print(f"Max batch size:      {max_batch:,}")
print(f"Mean batch size:     {statistics.mean(sizes):.2f}")
print(f"Median batch size:   {statistics.median(sizes):.2f}")
print(f"P10 batch size:      {percentile(sizes, 0.10):.2f}")
print(f"P90 batch size:      {percentile(sizes, 0.90):.2f}")
print(f"Min batch size:      {min(sizes):,}")
print(f"Max observed:       {max(sizes):,}")
print(
    f"Full batches:        {sum(x == max_batch for x in sizes) / len(sizes) * 100:.2f}%"
)
print(f"Mean fill:           {statistics.mean(sizes) / max_batch * 100:.2f}%")

# ------------------------------------------------------------------
# Per-executor batch statistics
# ------------------------------------------------------------------

print()
print("-" * 72)
print("PER EXECUTOR")
print("-" * 72)

by_exec = defaultdict(list)

for x in batches:
    by_exec[x["exec"]].append(x)

for executor_id in sorted(by_exec):
    xs = by_exec[executor_id]
    ss = [x["size"] for x in xs]

    print(f"Executor {executor_id}:")
    print(f"  batches:      {len(xs):,}")
    print(f"  mean batch:   {statistics.mean(ss):.2f}")
    print(f"  median batch: {statistics.median(ss):.2f}")
    print(f"  p10 batch:    {percentile(ss, 0.10):.2f}")
    print(f"  p90 batch:    {percentile(ss, 0.90):.2f}")
    print(f"  mean fill:    {statistics.mean(ss) / max_batch * 100:.2f}%")
    print(f"  full batches: {sum(x == max_batch for x in ss) / len(ss) * 100:.2f}%")

# ------------------------------------------------------------------
# Evaluation timing
# ------------------------------------------------------------------

if evals:
    eval_times = [x["eval_ms"] for x in evals]

    print()
    print("-" * 72)
    print("INFERENCE TIME")
    print("-" * 72)

    print(fmt(stats(eval_times)))
    print(f"Total eval time: {sum(eval_times) / 1000:.3f}s")

    by_eval_exec = defaultdict(list)

    for x in evals:
        by_eval_exec[x["exec"]].append(x["eval_ms"])

    for executor_id in sorted(by_eval_exec):
        print(f"Executor {executor_id}: {fmt(stats(by_eval_exec[executor_id]))}")

# ------------------------------------------------------------------
# Inter-batch gaps
# ------------------------------------------------------------------

timed_batches = [x for x in batches if x["t"] is not None]

if len(timed_batches) >= 2:
    print()
    print("-" * 72)
    print("INTER-BATCH GAPS")
    print("-" * 72)

    gaps_all = []
    gaps_by_exec = defaultdict(list)

    last_by_exec = {}

    for x in timed_batches:
        e = x["exec"]

        if e in last_by_exec:
            gap = x["t"] - last_by_exec[e]
            gaps_by_exec[e].append(gap)
            gaps_all.append(gap)

        last_by_exec[e] = x["t"]

    print(f"Overall: {fmt(stats(gaps_all))}")

    for executor_id in sorted(gaps_by_exec):
        print(f"Executor {executor_id}: {fmt(stats(gaps_by_exec[executor_id]))}")

# ------------------------------------------------------------------
# First vs last 10%
# ------------------------------------------------------------------

print()
print("-" * 72)
print("START vs END")
print("-" * 72)

n = len(batches)
window = max(1, n // 10)

first = batches[:window]
last = batches[-window:]

first_sizes = [x["size"] for x in first]
last_sizes = [x["size"] for x in last]

print(f"Window: {window:,} batches")
print()
print(
    f"First 10%: mean batch={statistics.mean(first_sizes):.2f}, "
    f"fill={statistics.mean(first_sizes) / max_batch * 100:.2f}%"
)
print(
    f"Last 10%:  mean batch={statistics.mean(last_sizes):.2f}, "
    f"fill={statistics.mean(last_sizes) / max_batch * 100:.2f}%"
)

if evals:
    first_evals = evals[: max(1, len(evals) // 10)]
    last_evals = evals[-max(1, len(evals) // 10) :]

    first_eval_times = [x["eval_ms"] for x in first_evals]
    last_eval_times = [x["eval_ms"] for x in last_evals]

    print(f"First 10%: mean eval={statistics.mean(first_eval_times):.3f} ms")
    print(f"Last 10%:  mean eval={statistics.mean(last_eval_times):.3f} ms")

# ------------------------------------------------------------------
# Overall throughput from timestamps
# ------------------------------------------------------------------

if timed_batches:
    start = timed_batches[0]["t"]
    end = timed_batches[-1]["t"]

    total_evals = sum(x["size"] for x in timed_batches)

    elapsed = end - start

    print()
    print("-" * 72)
    print("TIMELINE")
    print("-" * 72)

    print(f"Trace start:          {start:.3f}s")
    print(f"Trace end:            {end:.3f}s")
    print(f"Elapsed:              {elapsed:.3f}s")
    print(f"Evaluations:          {total_evals:,}")

    if elapsed > 0:
        print(f"Effective throughput: {total_evals / elapsed:,.1f} eval/s")

print()
print("=" * 72)
