#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import subprocess
import resource
import csv
import re
import argparse
from pathlib import Path
from datetime import datetime

# =========================
# 配置: 求解器定义与正则规则
# =========================
SOLVERS = {
    "d4": {
        "cmd": ["./d4", "-mc", "{file}"],
        "time_re": re.compile(r"c\s+Final time:\s+([\d.]+)"),
        "sol_re": re.compile(r"^s\s+(\d+)", re.MULTILINE),
        "decision_re": re.compile(r"c\s+Number of decision:\s+(\d+)")
    },
    "sharpsat-td": {
        "cmd": ["./sharpSAT", "-decot", "0.01", "-decow", "100", "-tmpdir", ".", "-cs", "3500", "{file}"],
        "time_re": re.compile(r"c\s+o\s+Solved in\s+([\d.]+)\s+seconds"),
        "sol_re": re.compile(r"c\s+s\s+exact arb int\s+(-?\d+)")
    },
    "exactMC": {
        "cmd": ["./exactMC", "ExactMC", "{file}"],
        "time_re": re.compile(r"Total time cost:\s+([\d.]+)"),
        "sol_re": re.compile(r"Number of models:\s+(-?\d+)")
    },
    "ganak": {
        "cmd": ["./ganak", "--mode", "0", "--appmct", "60", "--arjun", "0", "--puura", "0", "--vivif", "0", "--td", "0", "{file}"],
        "time_re": re.compile(r"c\s+o\s+Total time \[Arjun\+GANAK\]:\s*([\d.]+)"),
        "sol_re": re.compile(r"c\s+s\s+(?:exact|approx)\s+arb\s+int\s+(-?\d+)"),
        "sol_re_fallback": re.compile(r"c\s+o\s+intermediate count:\s*(-?\d+)")
    },
    "dmc": {
        "cmd": ["mpirun", "-np", "4", "./DeMoniaC", "-dmc", "{file}"],
        "time_re": re.compile(r"c\s+\[WORKER\s+\d+\]\s+Final time:\s+([\d.]+)"),
        "sol_re": re.compile(r"^s\s+(-?\d+)", re.MULTILINE)
    }
}

# INPUT_FOLDERS = ["data/bitwise", "data/pairwise", "data/ladder", "data/matrix"]
INPUT_FOLDERS = ["data/matrix"]

# 预期的编码类型（对应文件夹名称）
ENCODINGS = ["matrix"]

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """移除 ANSI 颜色/控制字符，避免正则匹配失败"""
    return ANSI_ESCAPE_RE.sub("", text)


def format_time(val: str):
    """将时间字符串格式化为 3 位小数；失败则返回 None"""
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return None


def parse_output(output: str, solver_name: str):
    """根据求解器规则解析控制台输出"""
    rules = SOLVERS[solver_name]
    output = strip_ansi(output)

    time_val = None
    sol_val = None
    decisions_val = None

    # decisions: 目前只有 d4 解析
    if solver_name == "d4" and "decision_re" in rules:
        matches = rules["decision_re"].findall(output)
        if matches:
            decisions_val = matches[-1]

    # time
    time_matches = rules["time_re"].findall(output)
    if time_matches:
        if solver_name == "dmc":
            try:
                time_val = f"{max(float(x) for x in time_matches):.3f}"
            except ValueError:
                time_val = None
        else:
            time_val = format_time(time_matches[-1])

    # solution
    sol_matches = rules["sol_re"].findall(output)
    if sol_matches:
        sol_val = sol_matches[-1]
    elif "sol_re_fallback" in rules:
        fallback_matches = rules["sol_re_fallback"].findall(output)
        if fallback_matches:
            sol_val = fallback_matches[-1]

    return time_val, sol_val, decisions_val

def limit_memory():
    """设置子进程虚拟内存上限为 15GB"""
    # 15 GB 转换为 Bytes
    max_memory_bytes = 15 * 1024 * 1024 * 1024 
    # 设置虚拟内存地址空间大小限制 (软限制, 硬限制)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

def run_solver(solver_name: str, filepath: str, timeout: int):
    """执行单个求解器并返回结果"""
    cmd_template = SOLVERS[solver_name]["cmd"]
    cmd = [part.format(file=filepath) for part in cmd_template]

    print(f"  执行: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            preexec_fn=limit_memory
        )

        output = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            if "Memory" in output or "bad_alloc" in output or result.returncode < 0:
                print(f"  [内存超出或崩溃] returncode={result.returncode}，可能超出了 15GB 限制。")
                return "MEMOUT", "MEMOUT", "MEMOUT"
            
        time_val, sol_val, decisions_val = parse_output(output, solver_name)

        if time_val is None and sol_val is None:
            print(f"  [解析失败] returncode={result.returncode}")
            return "ERR", "ERR", "ERR"

        return time_val or "ERR", sol_val or "ERR", decisions_val or "-"

    except subprocess.TimeoutExpired:
        print(f"  [超时] 超过 {timeout} 秒限制。")
        return "TO", "TO", "TO"
    except Exception as e:
        print(f"  [错误] 求解器运行异常: {e}")
        return "ERR", "ERR", "ERR"


def scan_instances(input_folders):
    """扫描文件夹，提取所有 instance，并与它们的不同编码路径绑定"""
    instance_map = {}

    for folder in input_folders:
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"[警告] 输入文件夹 {folder} 不存在或不是目录，跳过。")
            continue

        encoding = folder_path.name

        for file_path in folder_path.glob("*.cnf"):
            instance_name = file_path.stem
            if instance_name not in instance_map:
                instance_map[instance_name] = {}
            instance_map[instance_name][encoding] = str(file_path.absolute())

    return instance_map


def build_sorted_encodings(instance_map):
    """确定输出 CSV 的编码列顺序"""
    detected_encodings = set()
    for enc_dict in instance_map.values():
        detected_encodings.update(enc_dict.keys())

    # sorted_encodings = [e for e in ENCODINGS if e in detected_encodings]
    # for e in sorted(detected_encodings):
    #     if e not in sorted_encodings:
    #         sorted_encodings.append(e)

    return sorted(list(detected_encodings))


# =========================
# 追加模式相关辅助函数
# =========================
def find_existing_csv(output_dir: str, solver_name: str):
    """查找指定求解器最新的已存在 CSV 文件
    文件名按 `_` 分割：results_<solver>_<date>_<time>.csv
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    candidates = []
    for csv_file in output_path.glob("results_*.csv"):
        parts = csv_file.stem.split('_')
        if len(parts) >= 2 and parts[1] == solver_name:
            candidates.append(csv_file)

    if not candidates:
        return None
    # 按文件名排序（时间戳保证最新者在末尾）
    return sorted(candidates)[-1]


def load_existing_csv(csv_path: Path):
    """读取已存在 CSV 的 headers 和已完成实例集合"""
    completed = set()
    headers = None
    if not csv_path.exists():
        return headers, completed

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return headers, completed
        for row in reader:
            if row and row[0]:
                completed.add(row[0])
    return headers, completed


def parse_encodings_from_headers(headers):
    """从 CSV headers 解析出编码顺序"""
    if not headers:
        return []
    encs = []
    for h in headers[1:]:
        if h.endswith('_Time'):
            enc = h[:-len('_Time')]
            if enc not in encs:
                encs.append(enc)
    return encs


# =========================
# 实例处理 (含 -c 编码模式)
# =========================
def process_instance(solver_name, instance_map, instance, sorted_encodings,
                     encoding_mode, timeout):
    """处理单个实例的所有编码列，返回长度 = 3 * len(sorted_encodings) 的列表"""
    row_segments = []

    if encoding_mode == 'random':
        available = [e for e in sorted_encodings if e in instance_map[instance]]
        if not available:
            return ['-'] * (3 * len(sorted_encodings))

        chosen = random.choice(available)
        print(f"    [random] 选中编码: {chosen}")
        
        # 决定运行顺序：先随机选中的，再剩余的
        run_order = [chosen] + [e for e in available if e != chosen]

        results = {}            # enc -> (t, s, d)
        timed_out = False

        for enc in run_order:
            if timed_out:
                # 已经有编码超时，剩余未跑的统一记 TO，节省时间
                results[enc] = ('TO', 'TO', 'TO')
                print(f"    [random] 跳过 {enc}（前序已超时，记 TO）")
                continue

            t, s, d = run_solver(solver_name, instance_map[instance][enc], timeout)
            results[enc] = (t, s, d)
            if t == 'TO':
                timed_out = True
        
        # 按 sorted_encodings 列顺序输出
        for enc in sorted_encodings:
            if enc in results:
                row_segments.extend(results[enc])
            else:
                # 该编码文件不存在
                row_segments.extend(['-', '-', '-'])
        return row_segments

    elif encoding_mode in ENCODINGS:
        # 指定单一编码：仅该编码运行，其余列填 '-'
        for enc in sorted_encodings:
            if enc == encoding_mode:
                fp = instance_map[instance].get(enc)
                if fp:
                    t, s, d = run_solver(solver_name, fp, timeout)
                    row_segments.extend([t, s, d])
                else:
                    row_segments.extend(['-', '-', '-'])
            else:
                row_segments.extend(['-', '-', '-'])
        return row_segments

    else:  # 'all' 默认：四种全部跑
        for enc in sorted_encodings:
            fp = instance_map[instance].get(enc)
            if fp:
                t, s, d = run_solver(solver_name, fp, timeout)
                row_segments.extend([t, s, d])
            else:
                row_segments.extend(['-', '-', '-'])
        return row_segments


def main():
    parser = argparse.ArgumentParser(description="批量运行 Model Counters")
    parser.add_argument(
        "-i", "--input_folders",
        nargs="+",
        default=INPUT_FOLDERS,
        help="包含 CNF 文件的输入文件夹列表，例: -i data/matrix"
    )
    parser.add_argument('-o', '--output_dir', type=str, default="./results",
                        help="CSV 结果保存的输出目录")
    parser.add_argument('-t', '--timeout', type=int, default=1500,
                        help="单次求解的超时时间（秒），默认 1500")
    parser.add_argument('-s', '--solver', type=str, default=None,
                        choices=list(SOLVERS.keys()),
                        help="指定运行单个求解器，默认全部运行（方便开多进程并行不同求解器）")
    parser.add_argument('-a', '--append', action='store_true',
                        help="追加模式：在已存在的 CSV 上追加，跳过已记录的实例")
    parser.add_argument('-c', '--encoding', type=str, default='all',
                        help="编码方式: 'all'(默认四种全跑) | 'random'(随机选一种，超时则其他统一记 TO) | 具体编码名 (bitwise/pairwise/ladder/matrix)")

    args = parser.parse_args()

    # 校验 -c 取值
    valid_encoding_modes = ['all', 'random'] + ENCODINGS
    if args.encoding not in valid_encoding_modes:
        parser.error(f"-c 参数无效: '{args.encoding}'，应为 {valid_encoding_modes} 之一")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 50)
    print(" 开始扫描测试实例...")
    instance_map = scan_instances(args.input_folders)
    instances = sorted(instance_map.keys())
    print(f" 扫描完成，共找到 {len(instances)} 个唯一实例。")
    print("=" * 50)

    if not instances:
        print("[错误] 没有找到任何 .cnf 实例，程序结束。")
        return

    default_sorted_encodings = build_sorted_encodings(instance_map)

    # 选择要跑的求解器
    solvers_to_run = [args.solver] if args.solver else list(SOLVERS.keys())

    print(f" 求解器: {', '.join(solvers_to_run)}")
    print(f" 编码模式: {args.encoding}")
    print(f" 追加模式: {'开启' if args.append else '关闭'}")
    print("=" * 50)

    for solver_name in solvers_to_run:
        print(f"\n>>> 正在测试求解器: {solver_name}")

        # 决定 CSV 路径、表头与编码列顺序
        sorted_encodings = default_sorted_encodings
        existing_csv = None
        completed_instances = set()
        write_header = True

        if args.append:
            existing_csv = find_existing_csv(args.output_dir, solver_name)
            if existing_csv:
                existing_headers, completed_instances = load_existing_csv(existing_csv)
                csv_filename = str(existing_csv)
                if existing_headers:
                    existing_encs = parse_encodings_from_headers(existing_headers)
                    if existing_encs:
                        # 保持原有列顺序，避免错位
                        sorted_encodings = existing_encs
                    write_header = False
                    print(f"  [追加] 已存在 CSV: {csv_filename}")
                    print(f"  [追加] 编码列顺序沿用旧文件: {sorted_encodings}")
                    print(f"  [追加] 已完成 {len(completed_instances)} 条记录，将跳过。")

                    # 提示新检测到但旧文件没有的编码（会被丢弃）
                    missing_in_old = [e for e in default_sorted_encodings if e not in sorted_encodings]
                    if missing_in_old:
                        print(f"  [追加][警告] 新检测到的编码 {missing_in_old} 不在旧 CSV 列中，将被忽略。")
                else:
                    print(f"  [追加] CSV 文件 {csv_filename} 为空，将写入表头。")
            else:
                csv_filename = os.path.join(
                    args.output_dir, f"results_{solver_name}_{timestamp}.csv")
                print(f"  [追加] 未找到该求解器的旧记录，新建 CSV: {csv_filename}")
        else:
            csv_filename = os.path.join(
                args.output_dir, f"results_{solver_name}_{timestamp}.csv")

        # 准备表头
        headers = ["Instance"]
        for enc in sorted_encodings:
            headers.extend([f"{enc}_Time", f"{enc}_Solutions", f"{enc}_Decisions"])

        file_mode = 'a' if (args.append and existing_csv) else 'w'

        with open(csv_filename, file_mode, encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(headers)

            for idx, instance in enumerate(instances, 1):
                if instance in completed_instances:
                    print(f"  [{idx}/{len(instances)}] 跳过已完成: {instance}")
                    continue

                print(f"  [{idx}/{len(instances)}] 实例: {instance}")
                row_data = [instance]
                row_data.extend(
                    process_instance(solver_name, instance_map, instance,
                                     sorted_encodings, args.encoding, args.timeout)
                )
                writer.writerow(row_data)
                csvfile.flush()

        print(f"[*] 求解器 {solver_name} 的结果已保存至: {csv_filename}")


if __name__ == "__main__":
    main()