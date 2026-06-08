"""
NanoLM CLI — 项目总入口

用法:
    # 数据处理
    python -m toolkit full                 # 完整管线
    python -m toolkit download             # 仅下载
    python -m toolkit shuffle              # 仅打乱
    ...

    # 训练
    python -m toolkit train pretrain       # 启动预训练
    python -m toolkit train sft            # 启动 SFT 训练

    # 分析
    python -m toolkit analyze loss         # 绘制 loss 曲线
    python -m toolkit analyze lr           # 绘制学习率曲线

    # 评估
    python -m toolkit eval compare         # SFT vs PPO 对比
    python -m toolkit eval extract         # 提取 PPO policy 权重

    # Web 服务
    python -m toolkit web                  # 启动 Web 推理服务

    # 其他
    python -m toolkit list                 # 列出所有阶段
"""

import argparse
import sys

from toolkit.pipeline import STAGES, DataPipeline


def main():
    parser = argparse.ArgumentParser(
        description='NanoLM CLI',
        prog='python -m toolkit',
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # === 数据处理 ===

    full_parser = subparsers.add_parser('full', help='运行完整数据预处理管线')
    full_parser.add_argument('--skip-quality', action='store_true', help='跳过后端质量校验')
    full_parser.add_argument('--force', action='store_true', help='强制重新运行所有阶段')
    full_parser.add_argument('--seed', type=int, default=None, help='打乱种子（默认 42)')

    for name, desc in STAGES:
        subparsers.add_parser(name, help=desc)

    # === 训练 ===

    train_parser = subparsers.add_parser('train', help='启动训练')
    train_parser.add_argument('stage', choices=['pretrain', 'midtrain', 'sft', 'ppo'], help='训练阶段')
    train_parser.add_argument('-c', '--config', default=None, help='自定义 YAML 配置文件路径')

    # === 分析 ===

    analyze_parser = subparsers.add_parser('analyze', help='训练日志可视化')
    analyze_parser.add_argument('type', choices=['loss', 'lr'], help='分析类型')
    analyze_parser.add_argument('-f', '--file', dest='log_path', default=None, help='日志文件路径')

    # === 评估 ===

    eval_parser = subparsers.add_parser('eval', help='模型评估与后处理')
    eval_parser.add_argument('action', choices=['compare', 'extract'],
                             help='compare: SFT vs PPO 对比 | extract: 提取 PPO policy 权重')

    # === Web 服务 ===

    subparsers.add_parser('web', help='启动 Web 推理服务')

    # === 其他 ===

    subparsers.add_parser('list', help='列出所有阶段')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'list':
        DataPipeline.list_stages()

    elif args.command == 'train':
        from toolkit.train import run_training
        run_training(args.stage, args.config)

    elif args.command == 'analyze':
        from toolkit.analyze import run_analysis
        run_analysis(args.type, args.log_path)

    elif args.command == 'eval':
        if args.action == 'compare':
            from eval.compare import compare_sft_ppo
            compare_sft_ppo()
        elif args.action == 'extract':
            from eval.extract import extract_policy
            extract_policy()

    elif args.command == 'web':
        from nanolm.server import main as serve
        serve()

    elif args.command == 'full':
        pipeline = DataPipeline(
            force=args.force,
            skip_quality=args.skip_quality,
            seed=args.seed,
        )
        pipeline.run_full()

    else:
        pipeline = DataPipeline(
            force=getattr(args, 'force', False),
            skip_quality=getattr(args, 'skip_quality', False),
            seed=getattr(args, 'seed', None),
        )
        pipeline.run_stages([args.command])


if __name__ == '__main__':
    main()
