import re
import argparse
import pandas as pd

PUNCT_ANY = r"[.,!?;:\u3002\uFF0C\uFF1F\uFF01\uFF1B\uFF1A]"  # . , ? ! ; : + 中文 。 ， ？ ！ ； ：
re_any = re.compile(PUNCT_ANY)

def main(csv_path: str, col: str, chunksize: int, max_print: int, stop_after_print: bool):
    printed = 0
    total = 0
    matched = 0

    # 为了给出“全局行号”，我们自己维护一个 row_id（从 0 开始）
    row_id = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if col not in chunk.columns:
            raise ValueError(f"Column '{col}' not found. Available columns: {list(chunk.columns)[:50]}")

        # 不要丢 index，按 chunk 顺序遍历，自己累计 row_id
        vals = chunk[col].tolist()
        for v in vals:
            cur_row = row_id
            row_id += 1

            if not isinstance(v, str) or not v.strip():
                continue

            total += 1
            text = v.strip()
            if re_any.search(text):
                matched += 1
                if printed < max_print:
                    # 打印：全局行号 + prompt
                    print(f"[row={cur_row}] {text}")
                    printed += 1
                    if stop_after_print and printed >= max_print:
                        print(f"\n[STOP] printed {printed} matches, stop early.")
                        print(f"[SUMMARY] seen_valid_prompts={total}, matched={matched}")
                        return

    print(f"\n[SUMMARY] seen_valid_prompts={total}, matched={matched}, printed={printed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default="prompt")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--max_print", type=int, default=2000, help="最多打印多少条带标点的 prompt")
    ap.add_argument("--stop_after_print", action="store_true", help="打印够 max_print 后就提前结束（更快）")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        col=args.col,
        chunksize=args.chunksize,
        max_print=args.max_print,
        stop_after_print=args.stop_after_print,
    )