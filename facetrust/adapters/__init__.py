"""
facetrust.adapters
=====================

Integrações com implementações de terceiros (SSLWatermarking, RoSTEALS)
usadas no tutorial. Cada submódulo aqui é opcional -- requer que o
respectivo repositório de terceiros esteja clonado e (idealmente) suas
dependências (torch, etc.) instaladas via `pip install -e ".[ssl-watermarking]"`
ou `.[rosteals]` (ver pyproject.toml).

Diferente do resto do pacote `facetrust` (que só depende de numpy/scipy/
scikit-learn/scikit-image/pandas/matplotlib/Pillow), os módulos aqui
importam código de terceiros em tempo de execução (via `sys.path`), então
o pacote `facetrust` continua funcionando normalmente mesmo se você nunca
usar nada deste subpacote.
"""
