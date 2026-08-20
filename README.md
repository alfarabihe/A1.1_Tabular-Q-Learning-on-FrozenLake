# Q-learning Tabular di FrozenLake

<div align="center">

[![License](https://img.shields.io/github/license/alfarabihe/Ising-Model-for-Public-Opinion.svg?color=brightgreen)](https://github.com/alfarabihe/A1.1_Tabular-Q-Learning-on-FrozenLake/blob/master/LICENSE)
[![X](https://img.shields.io/badge/X-Share-black?logo=x)](https://twitter.com/intent/tweet?text=Ising-Model-for-Public-Opinion%20&url=https://github.com/alfarabihe/Ising-Model-for-Public-Opinion&hashtags=IsingModel,Simulation) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Falfarabihe%2FIsing-Model-for-Public-Opinion)

</div>

Implementasi Q-learning tabular (tanpa neural network) pada environment `FrozenLake-v1`
dari `gymnasium`, sebagai studi fondasi sebelum mempelajari algoritma Deep Q-Network (DQN)
seperti yang dipakai pada `ReinforcementAgent` di repo [**Stock-MARL**](https://github.com/peiyan03/Stock-MARL).

## Tujuan

Memahami secara eksplisit **Bellman equation** dan **update rule Q-learning**:

```
Q[s,a] += α × (r + γ × max(Q[s']) − Q[s,a])
```

dengan mengimplementasikannya langsung sebagai tabel `numpy`, tanpa lapisan abstraksi
neural network — sehingga setiap komponen rumus (reward, discount, TD-error) terlihat
jelas dalam kode.

## Isi Repo

| File | Deskripsi |
|---|---|
| [`qlearning_frozenlake.ipynb`](./qlearning_frozenlake.ipynb) | Notebook utama: teori (MDP, Bellman equation, TD-update, ε-greedy), implementasi Q-table, training loop, evaluasi policy, dan kaitannya dengan `DQN(...)` pada Stock-MARL. |
| [`output.png`](./output.png) | Kurva belajar (success rate per 100 episode) hasil training 8000 episode. |
| [`interpretasi_qlearning_frozenlake.md`](./interpretasi_qlearning_frozenlake.md) | Interpretasi naratif atas output aktual notebook — progres training, pembacaan kurva, evaluasi policy, dan analisis Q-table/policy akhir. |

## Ringkasan Hasil

- **Environment:** `FrozenLake-v1`, peta 4x4, `is_slippery=True` (stochastic), 16 state × 4 aksi.
- **Training:** 8000 episode, `alpha=0.1`, `gamma=0.99`, ε didecay eksponensial dari 1.0 → 0.01.
- **Success rate (evaluasi policy greedy, 1000 episode):** **72.9%** — memenuhi target 70–80%.
- Kurva belajar naik dari mendekati 0% di awal training menuju rentang target seiring ε menurun.

Detail lengkap beserta pembacaan Q-table dan policy akhir ada di
[`interpretasi_qlearning_frozenlake.md`](./interpretasi_qlearning_frozenlake.md).

## Cara Menjalankan

```bash
pip install gymnasium numpy matplotlib
jupyter notebook qlearning_frozenlake.ipynb
```

Jalankan seluruh cell secara berurutan. Notebook bersifat self-contained — tidak
memerlukan dependensi atau file eksternal lain.

## Kaitan dengan Repo [Stock-MARL](https://github.com/peiyan03/Stock-MARL) (`main.py`)

Notebook ini adalah versi "sebelum Deep" dari algoritma yang sama dipakai untuk melatih
`ReinforcementAgent` pada simulasi multi-agent pasar saham di repo [Stock-MARL](https://github.com/peiyan03/Stock-MARL):

| | Q-learning tabular (repo ini) | DQN (`main.py`, Stock-MARL) |
|---|---|---|
| Representasi $Q(s,a)$ | Tabel `numpy` eksplisit | Neural network (`MlpPolicy`) |
| State space | Diskrit, kecil (16 state) | Kontinu, berdimensi tinggi (fitur pasar) |
| Eksplorasi | ε-greedy dengan decay manual | ε-greedy (`exploration_fraction`, `exploration_final_eps`) |
| Stabilisasi | Tidak diperlukan | Replay buffer & target network |
| Update dasar | $Q(s,a) \mathrel{+}= \alpha(r+\gamma\max_{a'}Q(s',a')-Q(s,a))$ | TD-error yang sama, dioptimasi via gradient descent |

Memahami update rule di repo ini membuat parameter DQN pada `main.py` (`target_update_interval`,
`exploration_fraction`, `buffer_size`, dst.) lebih mudah dipahami secara konseptual.

## Lisensi

Project ini dilisensikan di bawah [MIT License](LICENSE).
