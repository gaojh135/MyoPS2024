import shutil
from pathlib import Path

train_data = {"CenterB": Path("./data/data_raw/CARE2024_MyoPS++/aligned_train/CenterB"),
              "CenterC": Path("./data/data_raw/CARE2024_MyoPS++/aligned_train/CenterC")}
test_data = Path("./data/data_raw/CARE2024_MyoPS++_valid/aligned/")

train_dir = Path("./data/data_preprocessed/train")
test_dir  = Path("./data/data_preprocessed/test")

exclude_cases = {"CenterB": {"Case2002", "Case2004", "Case2006", "Case2012", "Case2013", "Case2014", "Case2021", "Case2022", "Case2025", "Case2041", "Case2044"},
                 "CenterC": {"Case3008", "Case3009", "Case3023", "Case3037"}}

def copy_train_cases(sources, dest, exclude_map):
    dest.mkdir(parents=True, exist_ok=True)
    for center, src_dir in sources.items():
        print(f"Copying training cases from {center} ...")
        for case_dir in src_dir.iterdir():
            if not case_dir.is_dir():
                continue
            if case_dir.name in exclude_map.get(center, set()):
                print(f"  → Skip {case_dir.name}")
                continue
            dst = dest / case_dir.name
            shutil.copytree(case_dir, dst, dirs_exist_ok=True)
            print(f"  → Copied {case_dir.name}")

def copy_test_cases(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Copying test cases from {src.name} ...")
    for case_dir in src.iterdir():
        if not case_dir.is_dir():
            continue
        dst = dest / case_dir.name
        shutil.copytree(case_dir, dst, dirs_exist_ok=True)
        print(f"  → Copied {case_dir.name}")

if __name__ == "__main__":
    copy_train_cases(train_data, train_dir, exclude_cases)
    copy_test_cases(test_data, test_dir)
    print("Data copying complete.")
