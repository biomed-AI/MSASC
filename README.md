

## 三数据集

### Accuracy

| method  | main            | model            | source_1        | source_2   | target   | accuracy |
| ------- | --------------- | ---------------- | --------------- | ---------- | -------- | -------- |
| darn    | main.py         | model.py         | 10x_Chromium_v3 | Smart-seq2 | CEL-Seq2 | 0.857414 |
| pseudo  | main_pseudo.py  | model_pseudo.py  | 10x_Chromium_v3 | Smart-seq2 | CEL-Seq2 | 0.868821 |
| scAdapt | main_scAdapt.py | model_scAdapt.py | 10x_Chromium_v3 | Smart-seq2 | CEL-Seq2 | 0.585551 |

### UMAP

原始数据的 umap

source_1: 10x_Chromium_v3

source_2:   Smart-seq2

target: CEL-Seq2

![image-20220309103623584](img/image-20220309103623584.png)

---

method: darn

source_1: 10x_Chromium_v3

source_2:   Smart-seq2

target: CEL-Seq2

accuracy: 0.857414

![image-20220309103636166](img/image-20220309103636166.png)

---

method: pseudo

source_1: 10x_Chromium_v3

source_2:   Smart-seq2

target: CEL-Seq2

accuracy: 0.868821

![image-20220309183626548](img/image-20220309183626548.png)

---

method: scAdapt

source_1: 10x_Chromium_v3

source_2:   Smart-seq2

target: CEL-Seq2

accuracy: 0.585551

![image-20220311110102325](img/image-20220311110102325.png)

---

## 七数据集

### Accuracy

| method | main           | model           | Smart.seq2 | Seq.Well | inDrops  | Drop.seq | CEL.Seq2 | 10x_v2   | 10x_v3   | average accuracy |
| ------ | -------------- | --------------- | ---------- | -------- | -------- | -------- | -------- | -------- | -------- | ---------------- |
| darn   | main.py        | model.py        | 0.828897   | 0.751274 | 0.859052 | 0.829435 | 0.853612 | 0.924944 | 0.94103  | 0.855463429      |
| pseudo | main_pseudo.py | model_pseudo.py | 0.880228   | 0.755299 | 0.861027 | 0.830498 | 0.870722 | 0.928105 | 0.937927 | 0.866258         |

### UMAP

原始数据的 umap

source: [Smart.seq2, Seq.Well, inDrops, Drop.seq, 10x_v2, 10x_v3]

target: CEL.Seq2

![image-20220309105045289](img/image-20220309105045289.png)

---

method: darn

source: [Smart.seq2, Seq.Well, inDrops, Drop.seq, 10x_v2, 10x_v3]

target: CEL.Seq2

accuracy: 0.853612

![image-20220311102800882](img/image-20220311102800882.png)

---

method: pseudo

source: [Smart.seq2, Seq.Well, inDrops, Drop.seq, 10x_v2, 10x_v3]

target: CEL.Seq2

accuracy: 0.870722

![image-20220311103948050](img/image-20220311103948050.png)

---

## 四数据集

### Accuracy

注：mu 的修改是修改对抗部分 loss 的权重，默认情况下，mu = 0.01

| method                     | main               | model               | source_1    | source_2    | source_3 | target | accuracy |
| -------------------------- | ------------------ | ------------------- | ----------- | ----------- | -------- | ------ | -------- |
| darn                       | main.py            | model.py            | Baron_Human | Segerstolpe | Xin      | Muraro | 0.94409  |
| darn，mu 修改到 0.05       |                    |                     | Baron_Human | Segerstolpe | Xin      | Muraro | 0.746444 |
|                            |                    |                     |             |             |          |        |          |
| pseudo                     | main_pseudo.py     | model_pseudo.py     | Baron_Human | Segerstolpe | Xin      | Muraro | 0.948504 |
| pseudo，mu 修改到 0.05     |                    |                     | Baron_Human | Segerstolpe | Xin      | Muraro | 0.688082 |
| pseudo，对抗部分修改成 mmd | main_pseudo_mmd.py | model_pseudo_mmd.py | Baron_Human | Segerstolpe | Xin      | Muraro | 0.961256 |

---

| method                     | main               | model               | Muraro   | Segerstolpe | Xin      | Baron_Human | average accuracy |
| -------------------------- | ------------------ | ------------------- | -------- | ----------- | -------- | ----------- | ---------------- |
| pseudo，对抗部分修改成 mmd | main_pseudo_mmd.py | model_pseudo_mmd.py | 0.961256 | 0.840312    | 0.987265 | 0.861831    | 0.912666         |
| pseudo，对抗部分修改成 dann | main_pseudo_dann.py | model_pseudo_dann.py | 0.961746 | 0.834956    | 0.976542 | 0.898739    | 0.91799575         |

### UMAP

原始数据的 umap

source: [Segerstolpe, Xin, Baron_Human]

target: Muraro

![image-20220310142536997](img/image-20220310142536997.png)

---

method: pseudo

source: [Segerstolpe, Xin, Baron_Human]

target: Muraro

accuracy: 0.948504

![image-20220310145039216](img/image-20220310145039216.png)

---

method: pseudo，mu 修改到 0.05

source: [Segerstolpe, Xin, Baron_Human]

target: Muraro

accuracy: 0.688082

![image-20220310152650865](img/image-20220310152650865.png)

---

method: pseudo，对抗部分修改成 mmd

source: [Segerstolpe, Xin, Baron_Human]

target: Muraro

accuracy: 0.961256

![image-20220311113939963](img/image-20220311113939963.png)

---

method: pseudo，对抗部分修改成 dann

source: [Segerstolpe, Xin, Baron_Human]

target: Muraro

accuracy: 0.961746

![image-20220312155345843](img/image-20220312155345843.png)