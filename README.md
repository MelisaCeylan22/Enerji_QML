# EnerjiQML - Bitirme Projesi I (Starter Repo)

## Klasör Yapısı
- report/ : LaTeX rapor
- notebooks/ : EDA + preprocessing (Gün 3-4)
- src/ : Python pipeline kodları (Gün 1-7)
- data/raw , data/processed : veri ham ve işlenmiş
- outputs/ : doğrulama görselleri ve loglar

## 1) Ortam Kurulumu (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Qiskit Kurulum Doğrulama (Gün 1 kanıt)
```bash
python src/verify_qiskit.py
```

Çıktılar:
- outputs/hadamard_counts.png
- outputs/bell_counts.png
- outputs/parametric_counts.png

Aynı görseller otomatik olarak report/figs/ altına kopyalanır.

## 3) Rapor Derleme
### Overleaf
- report/ klasörünü proje olarak yükle.
- main.tex derle.

### Yerel
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
