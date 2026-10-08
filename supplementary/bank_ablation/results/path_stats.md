# Reasoning-path reuse and entropy (teacher training data)

path = tuple of step names; bank = path given to the teacher; used = keys of the written rationale; step = individual step names. reuse = 1 - distinct/n; share>=2 = fraction of items whose path occurs at least twice; H = Shannon entropy (bits); H/log2k = normalised; perplexity = 2^H; top1 = share of the most frequent path.

| dataset | arm | which | n | distinct | reuse | share>=2 | H (bits) | H/log2k | perplexity | top1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MATH | baseline | bank path | 10000 | 294 | 0.971 | 0.998 | 7.18 | 0.875 | 144.6 | 0.031 |
| MATH | baseline | used path | 9562 | 868 | 0.909 | 0.948 | 7.85 | 0.804 | 229.9 | 0.028 |
| MATH | baseline | bank step | 22190 | 514 | 0.977 | 0.999 | 7.94 | 0.882 | 246.2 | 0.020 |
| MATH | baseline | used step | 20242 | 1183 | 0.942 | 0.974 | 8.40 | 0.823 | 337.6 | 0.017 |
| MATH | empty | used path | 9473 | 5210 | 0.450 | 0.604 | 11.66 | 0.944 | 3236.2 | 0.010 |
| MATH | empty | used step | 12719 | 5614 | 0.559 | 0.717 | 11.53 | 0.925 | 2950.9 | 0.014 |
| MATH | random | bank path | 10000 | 603 | 0.940 | 0.990 | 8.82 | 0.955 | 452.8 | 0.005 |
| MATH | random | used path | 9599 | 2541 | 0.735 | 0.791 | 9.74 | 0.861 | 852.2 | 0.010 |
| MATH | random | bank step | 19859 | 706 | 0.964 | 1.000 | 9.24 | 0.976 | 603.2 | 0.009 |
| MATH | random | used step | 19132 | 2945 | 0.846 | 0.901 | 9.80 | 0.850 | 890.4 | 0.012 |
| MATH | randglobal | bank path | 10000 | 601 | 0.940 | 0.991 | 8.83 | 0.957 | 455.7 | 0.006 |
| MATH | randglobal | used path | 9581 | 2565 | 0.732 | 0.789 | 9.77 | 0.863 | 872.0 | 0.011 |
| MATH | randglobal | bank step | 19825 | 707 | 0.964 | 1.000 | 9.24 | 0.976 | 604.3 | 0.008 |
| MATH | randglobal | used step | 19087 | 3012 | 0.842 | 0.898 | 9.84 | 0.851 | 914.9 | 0.011 |
| MATH | filtered | bank path | 7482 | 283 | 0.962 | 0.997 | 7.13 | 0.875 | 140.0 | 0.031 |
| MATH | filtered | used path | 7482 | 700 | 0.906 | 0.951 | 7.74 | 0.819 | 213.2 | 0.028 |
| MATH | filtered | bank step | 16516 | 501 | 0.970 | 0.998 | 7.91 | 0.881 | 239.8 | 0.023 |
| MATH | filtered | used step | 15798 | 988 | 0.937 | 0.975 | 8.32 | 0.837 | 320.3 | 0.017 |
| AQUA | baseline | bank path | 10000 | 255 | 0.975 | 0.996 | 6.72 | 0.841 | 105.7 | 0.044 |
| AQUA | baseline | used path | 9922 | 435 | 0.956 | 0.982 | 7.16 | 0.816 | 142.5 | 0.038 |
| AQUA | baseline | bank step | 19368 | 419 | 0.978 | 0.997 | 7.51 | 0.862 | 182.7 | 0.023 |
| AQUA | baseline | used step | 18992 | 596 | 0.969 | 0.991 | 7.74 | 0.840 | 214.5 | 0.030 |
| AQUA | empty | used path | 9609 | 4495 | 0.532 | 0.643 | 10.90 | 0.898 | 1910.8 | 0.017 |
| AQUA | empty | used step | 11965 | 4598 | 0.616 | 0.731 | 10.72 | 0.881 | 1687.2 | 0.015 |
| AQUA | random | bank path | 10000 | 739 | 0.926 | 0.989 | 9.10 | 0.955 | 548.1 | 0.006 |
| AQUA | random | used path | 9875 | 1678 | 0.830 | 0.894 | 9.43 | 0.881 | 692.2 | 0.017 |
| AQUA | random | bank step | 18914 | 695 | 0.963 | 1.000 | 9.12 | 0.966 | 557.2 | 0.031 |
| AQUA | random | used step | 18812 | 1683 | 0.911 | 0.953 | 9.13 | 0.852 | 561.2 | 0.048 |
| AQUA | randglobal | bank path | 10000 | 732 | 0.927 | 0.990 | 9.11 | 0.958 | 554.2 | 0.004 |
| AQUA | randglobal | used path | 9876 | 1724 | 0.825 | 0.890 | 9.48 | 0.882 | 716.3 | 0.017 |
| AQUA | randglobal | bank step | 18898 | 695 | 0.963 | 1.000 | 9.15 | 0.969 | 566.5 | 0.030 |
| AQUA | randglobal | used step | 18792 | 1678 | 0.911 | 0.955 | 9.19 | 0.858 | 583.2 | 0.044 |
| AQUA | filtered | bank path | 8827 | 246 | 0.972 | 0.996 | 6.70 | 0.843 | 103.8 | 0.044 |
| AQUA | filtered | used path | 8827 | 410 | 0.954 | 0.982 | 7.13 | 0.822 | 140.3 | 0.037 |
| AQUA | filtered | bank step | 17098 | 405 | 0.976 | 0.997 | 7.49 | 0.865 | 179.7 | 0.023 |
| AQUA | filtered | used step | 16890 | 571 | 0.966 | 0.990 | 7.73 | 0.844 | 211.6 | 0.030 |
