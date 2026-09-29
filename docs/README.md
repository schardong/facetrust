# docs/ — página do tutorial (GitHub Pages)

Página estática (HTML + CSS puros, sem build). Para publicar:
Settings → Pages → Source: "Deploy from a branch" → branch `main`, pasta `/docs`.

## Pendências antes de publicar
- Copiar `slides.pdf` e `manuscript.pdf` para `docs/assets/`.
- Coautores/afiliações no `<header>` de `index.html`.
- Trocar `GITHUB_USER/GITHUB_REPO` no link "Open in Colab".
- Ajustar o BibTeX.

## Observações
- `data.zip` e `morphs.zip` são grandes demais para o GitHub (limite de 100 MB por arquivo), então
  a página aponta para o Google Drive (mesmos IDs usados no notebook).
- `assets/facetrust.zip` e `assets/facetrust_colab_demo.ipynb` são cópias dos
  arquivos da raiz; atualize-os se mudar o código ou o notebook.
- `.nojekyll` desativa o Jekyll (serve os arquivos como estão).
