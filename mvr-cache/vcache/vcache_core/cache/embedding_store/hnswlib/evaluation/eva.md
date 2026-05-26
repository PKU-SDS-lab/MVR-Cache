# evaluate
## 安装：
conda create -n eva python==3.10
conda activate eva
pip install numpy pytrec-eval
pip install packaging


-----
## 运行：
conda activate eva

# 修改：
GT_FILE_PATH = "./ground-truth/gt.npy"
RESULTS_FILE_PATH = "./result/multi-hnsw.tsv" # plaid

python evaluate.py


----
## 数据准备
- ground-truth：
gt.npy文件如下
```
[[999000],
 [999001],
 [999002],
 [999003],
 [999004],]
```

- result：
tsv文件如下
```
0	999000	1	13.585735321044922
0	825506	2	13.378536224365234
```