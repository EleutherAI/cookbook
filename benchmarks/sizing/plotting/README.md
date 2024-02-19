# Plotting

Various ways to plot benchmark results produced by [these tools](..).

## Automated plotting

This script can plot `mm_flops.py` and `bmm_flops.py` results automatically:
```
python plotting/bplot.py --results_file mm_m_range_0_20k_16_n2k_k2k-env-vars.txt --notes "MI300X F.linear made by mm_flops.py"
```

## Tweak the notebook

[transformer_figures.ipynb](transformer_figures.ipynb)
