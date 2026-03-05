import pandas as pd
import os
import argparse
import glob

def merge_results(base_filename, output_folder, num_parts=None, delete_parts=False):
    dfs = []
    
    # 查找所有匹配的分片文件
    pattern = os.path.join(output_folder, f"{base_filename}_part*-eval.csv")
    files = glob.glob(pattern)
    
    # 按 part 编号排序
    # 假设文件名格式为 *_partN-eval.csv
    try:
        files.sort(key=lambda x: int(x.split('_part')[-1].split('-eval')[0]))
    except Exception as e:
        print(f"Warning: Could not sort files numerically by part number. Sorting alphabetically. Error: {e}")
        files.sort()
        
    print(f"Found {len(files)} part files for {base_filename}")
    for f in files:
        print(f"Reading {f}...")
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return

    if not dfs:
        print("No files found to merge.")
        return

    # 合并
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(merged_df)} rows.")

    # 保存
    output_filename = f"{base_filename}-eval.csv"
    output_path = os.path.join(output_folder, output_filename)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged file to {output_path}")

    # 删除分片
    if delete_parts:
        print("Deleting part files...")
        for f in files:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_filename", type=str, required=True, help="Base filename (without extension)")
    parser.add_argument("-of", "--output_folder", type=str, default="./result/Language Style")
    parser.add_argument("--delete", action="store_true", help="Delete part files after merging")
    args = parser.parse_args()

    merge_results(args.base_filename, args.output_folder, delete_parts=args.delete)
