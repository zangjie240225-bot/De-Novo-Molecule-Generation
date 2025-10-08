
#生成10000个符合定义条件的小分子。用法python 10000_tuned_denovo.py
import os
import sys
sys.path.append(os.path.realpath('.'))

from time import time
import pandas as pd
from tdc import Oracle, Evaluator
from genmol.sampler import Sampler

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# ----------- 参数设置 ------------
TARGET_NUM = 10000   # 想要的合格分子数
BATCH_SIZE = 1000   # 每次生成多少个候选分子
# --------------------------------

def check_props(smiles):
    """检查分子是否符合 CNS-like 条件"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)

    if not (240 <= mw <= 300):
        return None
    if not (2 <= logp <= 3):
        return None
    if not (tpsa <= 70):
        return None
    if not (1 <=hbd <= 2):
        return None

    return {
        "MolWt": mw,
        "LogP": logp,
        "TPSA": tpsa,
        "HBD": hbd,
    }

if __name__ == '__main__':
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    sampler = Sampler('model.ckpt')

    results = []
    total_generated = 0

    t_start = time()
    while len(results) < TARGET_NUM:
        samples = sampler.de_novo_generation(BATCH_SIZE, softmax_temp=0.5, randomness=0.5)
        total_generated += len(samples)

        df = pd.DataFrame({
            'smiles': samples,
            'qed': oracle_qed(samples),
            'sa': oracle_sa(samples)
        })
        df = df.drop_duplicates('smiles')

        # 先 QED/SA 过滤
        df = df[(df['qed'] >= 0.6) & (df['sa'] <= 4)]

        # 再理化性质过滤
        for smi, qed, sa in zip(df['smiles'], df['qed'], df['sa']):
            props = check_props(smi)
            if props:
                results.append({
                    "smiles": smi,
                    "qed": round(qed, 2),
                    "sa": round(sa, 2),
                    **{k: round(v, 2) for k, v in props.items()}
                })
                if len(results) >= TARGET_NUM:
                    break

        print(f"✅ 已收集 {len(results)} / {TARGET_NUM}，累计生成 {total_generated} 分子")

    df_final = pd.DataFrame(results)
    df_final.to_csv("denovo_results_final.csv", index=False)
    print(f"\n�� 完成：收集到 {len(results)} 个合格分子，结果已保存到 denovo_results_final.csv")
    print(f"总耗时 {time() - t_start:.2f} 秒，累计生成 {total_generated} 分子")

