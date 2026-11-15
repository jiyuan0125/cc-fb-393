#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""温度读数异常检测：离群 / 卡死 / 漂移。

用法: python3 detect.py [readings.csv] [params.txt] [anomalies.csv]

- readings.csv  : 设备,时间,读数（多台设备混排，15 分钟一条）
- params.txt    : 三行，依次为 离群倍数 / 卡死最短时长(小时) / 漂移阈值
- anomalies.csv : 异常登记，重跑时按 (设备,类型,起,止) 去重，同一段不进两回
基线：每台设备前 7 日的统计（中位数 + 稳健σ）。卡死段读数不参与离群判断。
"""
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

TIME_FMT = "%Y-%m-%d %H:%M:%S"
INTERVAL_MIN = 15           # 采样间隔（分钟）
BASELINE_DAYS = 7           # 基线取前 7 日
DRIFT_NOISE = 0.05          # 日均值逐日噪声容限，小于此不算方向改变


def load_params(path):
    with open(path, encoding="utf-8") as f:
        vals = [float(line.strip()) for line in f if line.strip()]
    if len(vals) < 3:
        raise ValueError("params.txt 需要三行：离群倍数 / 卡死最短时长(小时) / 漂移阈值")
    return vals[0], vals[1], vals[2]


def load_readings(path):
    """返回 {设备: [(datetime, 读数), ...]}，按时间排序。"""
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = datetime.strptime(row["时间"].strip(), TIME_FMT)
            data[row["设备"].strip()].append((t, float(row["读数"])))
    for dev in data:
        data[dev].sort()
    return data


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def detect_stuck(rows, min_hours):
    """连续相同读数且时长 >= min_hours 的段。返回 [(起, 止, 值, 点数)]。"""
    min_pts = int(min_hours * 60 / INTERVAL_MIN) + 1  # 时长=(点数-1)*间隔
    segs = []
    i, n = 0, len(rows)
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1][1] == rows[i][1]:
            j += 1
        pts = j - i + 1
        if pts >= min_pts:
            segs.append((rows[i][0], rows[j][0], rows[i][1], pts))
        i = j + 1
    return segs


def detect_outliers(rows, baseline_end, med, sigma, mult, stuck_mask,
                    drift_level=None):
    """|读数-期望水平| > mult*sigma。卡死段跳过；漂移段内期望水平取当日日均。"""
    thr = mult * sigma
    flagged = []
    for idx, (t, v) in enumerate(rows):
        if t < baseline_end or stuck_mask[idx]:
            continue
        level = med
        if drift_level is not None and t.date() in drift_level:
            level = drift_level[t.date()]
        dev = abs(v - level)
        if dev > thr:
            flagged.append((t, v, dev / sigma if sigma > 0 else float("inf")))
    # 相邻（间隔一个采样周期）的离群点合并为一段
    segs = []
    for t, v, ratio in flagged:
        if segs and t - segs[-1][1] <= timedelta(minutes=INTERVAL_MIN):
            segs[-1][1] = t
            segs[-1][2].append((v, ratio))
        else:
            segs.append([t, t, [(v, ratio)]])
    return segs, thr


def detect_drift(rows, baseline_end, base_mean, drift_thr):
    """日均值同向连挪、累计位移 >= drift_thr 的段。
    返回 ([(起日, 止日, 位移, 末日均)], {漂移段内日期: 当日日均})。"""
    daily = defaultdict(list)
    for t, v in rows:
        daily[t.date()].append(v)
    days = sorted(daily)
    means = {d: sum(daily[d]) / len(daily[d]) for d in days}
    segs = []
    drift_level = {}
    i = 1
    while i < len(days):
        delta = means[days[i]] - means[days[i - 1]]
        if abs(delta) <= DRIFT_NOISE:
            i += 1
            continue
        direction = 1 if delta > 0 else -1
        j = i
        while j + 1 < len(days):
            d = means[days[j + 1]] - means[days[j]]
            if d * direction <= DRIFT_NOISE:  # 反向或停滞，段结束
                break
            j += 1
        start, end = days[i - 1], days[j]  # 段起 = 开始挪动的前一天
        displacement = means[end] - means[start]
        if abs(displacement) >= drift_thr and end >= baseline_end.date():
            segs.append((start, end, displacement, means[end]))
            for d in days:
                if start <= d <= end:
                    drift_level[d] = means[d]
        i = j + 1
    return segs, drift_level


def main():
    readings_path = sys.argv[1] if len(sys.argv) > 1 else "readings.csv"
    params_path = sys.argv[2] if len(sys.argv) > 2 else "params.txt"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "anomalies.csv"

    mult, stuck_hours, drift_thr = load_params(params_path)
    data = load_readings(readings_path)

    new_rows = []  # (设备, 类型, 起, 止, 判定依据)
    for dev in sorted(data):
        rows = data[dev]
        t0 = rows[0][0]
        baseline_end = t0 + timedelta(days=BASELINE_DAYS)
        base_vals = [v for t, v in rows if t < baseline_end]
        med = median(base_vals)
        sigma = 1.4826 * median([abs(v - med) for v in base_vals])
        base_mean = sum(base_vals) / len(base_vals)

        # 1) 卡死（先判，供离群剔除）
        stuck_mask = [False] * len(rows)
        for start, end, val, pts in detect_stuck(rows, stuck_hours):
            hours = (pts - 1) * INTERVAL_MIN / 60
            new_rows.append((dev, "卡死", start, end,
                             "读数=%.3f 持续=%.2f小时(%d个点)"
                             % (val, hours, pts)))
            for idx, (t, _) in enumerate(rows):
                if start <= t <= end:
                    stuck_mask[idx] = True

        # 2) 漂移（日均值同向缓挪；先于离群判，供离群用漂移后的期望水平）
        drift_segs, drift_level = detect_drift(rows, baseline_end,
                                               base_mean, drift_thr)
        for start, end, disp, end_mean in drift_segs:
            days = (end - start).days
            new_rows.append((dev, "漂移",
                             datetime(start.year, start.month, start.day),
                             datetime(end.year, end.month, end.day),
                             "基线日均=%.3f 段末日均=%.3f 累计漂移=%+.3f 速率=%+.3f/天"
                             % (base_mean, end_mean, disp, disp / days)))

        # 3) 离群（剔除卡死段；漂移段内相对当日日均判）
        out_segs, thr = detect_outliers(rows, baseline_end, med, sigma,
                                        mult, stuck_mask, drift_level)
        for start, end, pts in out_segs:
            worst = max(pts, key=lambda p: p[1])
            new_rows.append((dev, "离群", start, end,
                             "读数=%.3f 基线中位数=%.3f 阈值=%.3f(%.1fσ) 偏离=%.1fσ"
                             % (worst[0], med, thr, mult, worst[1])))

    # 幂等合并：读已有登记，按 (设备,类型,起,止) 去重
    header = ["设备", "异常类型", "起始时间", "结束时间", "判定依据"]
    existing = []
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing.append([row[h] for h in header])
    seen = {(r[0], r[1], r[2], r[3]) for r in existing}
    added = 0
    for dev, kind, start, end, basis in new_rows:
        rec = [dev, kind, start.strftime(TIME_FMT), end.strftime(TIME_FMT), basis]
        key = tuple(rec[:4])
        if key not in seen:
            seen.add(key)
            existing.append(rec)
            added += 1
    existing.sort(key=lambda r: (r[0], r[2], r[1]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(existing)

    # 汇总：三类各自的段数与涉及设备数
    print("本次新登记 %d 段，登记簿共 %d 段" % (added, len(existing)))
    print("%-6s %8s %8s" % ("类型", "段数", "设备数"))
    for kind in ("离群", "卡死", "漂移"):
        segs = [r for r in existing if r[1] == kind]
        devs = {r[0] for r in segs}
        print("%-6s %8d %8d" % (kind, len(segs), len(devs)))


if __name__ == "__main__":
    main()
