# 项目启动入口
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="SegmentationSystem Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        os.system("python scripts/train.py")
    elif args.mode == 'eval':
        os.system("python scripts/evaluate.py")


if __name__ == '__main__':
    main()
