# Fairness Impossibility Analysis

## Key correlations (Spearman)

|             |        EOD |   ECE_overall |   ECE_range |   ECE_std |
|:------------|-----------:|--------------:|------------:|----------:|
| EOD         |  1         |     -0.278568 |   0.0989016 | -0.013295 |
| ECE_overall | -0.278568  |      1        |   0.420367  |  0.495012 |
| ECE_range   |  0.0989016 |      0.420367 |   1         |  0.975043 |
| ECE_std     | -0.013295  |      0.495012 |   0.975043  |  1        |

## EO-enforced vs others (means)

| EO_enforced   |      EOD |   ECE_overall |   ECE_range |   ECE_std |
|:--------------|---------:|--------------:|------------:|----------:|
| False         | 0.387786 |    0.0973413  |    0.22535  | 0.0811242 |
| True          | 0.201153 |    0.00790706 |    0.198536 | 0.0608634 |

## Impossibility gap (calibration dispersion when pushing EOD low)

| metric                      | description                                                                    |      value |   low_eod_threshold |   high_eod_threshold |
|:----------------------------|:-------------------------------------------------------------------------------|-----------:|--------------------:|---------------------:|
| impossibility_gap_ECE_range | Mean calibration range (low EOD quartile) minus mean range (high EOD quartile) | -0.0114722 |            0.133163 |             0.666278 |

Notes:
- Slope(EOD -> ECE_range): 0.0282 (negative means tighter EO increases calibration spread if positive).
- Slope(EOD -> ECE_overall): -0.1499.
- Use the gap to flag whether driving EOD to lower quartile systematically raises calibration dispersion.
