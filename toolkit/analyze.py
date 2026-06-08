"""
训练日志可视化

用法:
    python -m toolkit analyze loss     # 绘制 loss 曲线
    python -m toolkit analyze loss -f ./my_log.txt
    python -m toolkit analyze lr       # 绘制学习率曲线
    python -m toolkit analyze lr -f ./my_lr.txt
"""

import math
import os
import re
import sys


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib 未安装。运行: pip3 install matplotlib")
        sys.exit(1)


def plot_loss(log_path='./log/log.txt'):
    """解析 log.txt，绘制所有指标曲线"""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        sys.exit(1)

    data_map = {}
    all_metric_keys = []

    with open(log_path) as f:
        for line in f:
            if '====' in line or '->' not in line:
                continue
            try:
                meta_part, values_part = line.split(' -> ')
                epoch_match = re.search(r'epoch:\s*(\d+)', meta_part)
                file_match = re.search(r'file:\s*(\d+)', meta_part)
                batch_match = re.search(r'batch:\s*(\d+)', meta_part)
                if not (epoch_match and file_match and batch_match):
                    continue

                epoch = int(epoch_match.group(1))
                file_idx = int(file_match.group(1))
                batch_idx = int(batch_match.group(1))
                sort_key = (epoch, file_idx, batch_idx)

                current_metrics = {}
                for kv in values_part.split(', '):
                    k, v = kv.split(': ')
                    val = float(v.strip())
                    current_metrics[k] = val
                    if k not in all_metric_keys:
                        all_metric_keys.append(k)

                data_map[sort_key] = current_metrics
            except Exception:
                continue

    if not data_map:
        print("日志中没有有效数据")
        sys.exit(1)

    sorted_keys = sorted(data_map.keys())
    results = {k: [] for k in all_metric_keys}
    separator_indices = []
    prev_key = sorted_keys[0]

    for i, key in enumerate(sorted_keys):
        metrics = data_map[key]
        for k in all_metric_keys:
            results[k].append(metrics.get(k, None))

        curr_epoch, curr_file, _ = key
        prev_epoch, prev_file, _ = prev_key
        if curr_epoch != prev_epoch:
            separator_indices.append((i, 'epoch', f'Ep {curr_epoch}'))
        elif curr_file != prev_file:
            separator_indices.append((i, 'file', f'F {curr_file}'))
        prev_key = key

    n_metrics = len(results)
    cols = 4 if n_metrics >= 4 else n_metrics
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    flat_axes = [axes] if n_metrics == 1 else axes.flatten()

    for idx, metric_name in enumerate(results.keys()):
        ax = flat_axes[idx]
        y = results[metric_name]
        x = range(len(y))

        ax.plot(x, y, linewidth=1.0, label=metric_name)

        for sep_idx, sep_type, sep_label in separator_indices:
            if sep_type == 'epoch':
                ax.axvline(x=sep_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                if idx == 0:
                    ax.text(sep_idx, ax.get_ylim()[1], sep_label,
                            rotation=90, verticalalignment='top', color='red', fontsize=8)
            elif sep_type == 'file':
                ax.axvline(x=sep_idx, color='green', linestyle=':', linewidth=1.0, alpha=0.6)
                if idx == 0:
                    ax.text(sep_idx, ax.get_ylim()[1], sep_label,
                            rotation=90, verticalalignment='top', color='green', fontsize=8)

        ax.set_title(metric_name)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel("Steps")
        ax.grid(True, linestyle='--', alpha=0.3)

    for i in range(n_metrics, len(flat_axes)):
        flat_axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_lr(log_path='./log/lr.txt'):
    """解析 lr.txt，绘制学习率曲线"""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        sys.exit(1)

    lrs = {}
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = line.split('step: ')[-1]
                data = data.split(', lr:')
                step = int(data[0].strip())
                lr = float(data[1].strip())
                lrs[step] = lr
            except (IndexError, ValueError):
                continue

    if not lrs:
        print("日志中没有有效数据")
        sys.exit(1)

    sorted_data = sorted(lrs.items(), key=lambda x: x[0])
    x = [item[0] for item in sorted_data]
    y = [item[1] for item in sorted_data]

    plt.figure(figsize=(10, 6))
    plt.title('Learning Rate')
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.plot(x, y, linewidth=1.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=30)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def run_analysis(analysis_type, log_path=None):
    """
    运行日志可视化。

    Args:
        analysis_type: 'loss' 或 'lr'
        log_path: 日志文件路径，None 则使用默认路径
    """
    defaults = {
        'loss': './log/log.txt',
        'lr':   './log/lr.txt',
    }

    if analysis_type not in defaults:
        print(f"未知分析类型: {analysis_type}")
        print(f"可用: {', '.join(defaults.keys())}")
        sys.exit(1)

    path = log_path or defaults[analysis_type]

    if analysis_type == 'loss':
        plot_loss(path)
    else:
        plot_lr(path)
