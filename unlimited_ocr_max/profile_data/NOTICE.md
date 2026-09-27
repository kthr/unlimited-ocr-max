# Attribution

Four of the twelve pages bundled under `profile_data/pages/` are 200-dpi
renders of pages of *Unlimited OCR Works* (Youyang Yin, Huanhuan Liu, YY,
Qunyi Xie, Chaorun Liu, Shiqi Yang, Shaohua Wang, Zhanlong Liu, Hao Zou,
Jinyue Chen, Shu Wei, Jingjing Wu, Mingxin Huang, Zhen Wu, Guibin Wang,
Tengyu Du, Lei Jia; 2026), [arXiv:2606.23050](https://arxiv.org/abs/2606.23050):

| file | paper page |
| --- | --- |
| `pages/figure_wide.png` | page 1 (Figure 1) |
| `pages/toc_dotted.png` | page 2 (table of contents) |
| `pages/dense_body.png` | page 3 (Introduction) |
| `pages/plain_text.png` | page 14 (references tail) |

The matching `references/bf16/{name}.md` and `references/int8/{name}.md` files
for those four pages are this model's own OCR output of those renders, not
text taken from the paper.

The same paper is distributed by Baidu as `Unlimited-OCR.pdf` in the
MIT-licensed `baidu/Unlimited-OCR` repository (MIT, Copyright (c) 2026 Baidu),
<https://huggingface.co/baidu/Unlimited-OCR>.

The remaining 8 `syn_*` pages, and all of their reference transcripts, are
this project's own synthetic pages.
