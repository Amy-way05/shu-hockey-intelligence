# Penalty-Kill Decision-Support Framework

**[Live Research Dashboard →](https://ncaa-pk-intelligence.streamlit.app)**
**[Published Preprint → SportRxiv](https://doi.org/10.51224/SportRxiv.972)**

A reinforcement learning and Markov Chain framework for quantifying penalty-kill deployment risk in NCAA Division I ice hockey, published as a first-author preprint with a six-author cross-institutional research team.

## Research Summary

When a hockey team is shorthanded due to a penalty, coaching staff must make real-time personnel deployment decisions with limited information and significant consequences. This study replaces instinct-driven decision-making with a data-driven framework built on 13,777 penalty outcomes across 55 NCAA Division I programmes.

Two complementary modeling approaches were applied:

**Q-Learning (Reinforcement Learning):** An 8-state reward structure trained over 10,000 iterations to learn which deployment actions minimize goal-against probability across different game contexts (period, score differential, pressure level). Policy agreement stabilizes at 0.94 to 0.96 within the study's event count, confirmed via bootstrap resampling.

**Markov Chain Transition Modeling:** State-transition probabilities capturing how penalty situations evolve across game states, providing interpretable probability estimates directly usable by coaching staff during live games.

## Validation

The framework was validated through three independent methods:

- **Leave-one-season-out (LOSO) cross-validation** across all available seasons, with calibration error below 0.05 for the optimal bin configuration
- **500-game bootstrap resampling** confirming that observed goal-against rates fall within bootstrapped confidence intervals for all eight game states
- **Cross-programme calibration** across all 55 Division I programmes, demonstrating the model generalizes beyond the study programme's own data (Wilcoxon p < 0.0001)

## Live Dashboard

The interactive Streamlit dashboard provides:

- Q-learning convergence analysis with confidence bands
- Markov state-conditioned reward exploration
- Filterable optimal policy table by period, score state, and pressure level
- LOSO and bootstrap validation results
- Cross-programme calibration distributions
- All five publication figures from the preprint

## Confidentiality

All player names, programme names, and conference affiliations have been anonymized throughout this repository and dashboard. No tactical strategy, individual performance data, or identifiable game film is disclosed. The published figures use "Study programme" in place of the actual institution name.

## Repository Structure

```
app.py                     Streamlit dashboard application
requirements.txt           Python dependencies
.streamlit/config.toml     Dashboard theme configuration
data/                      Anonymized model outputs and validation results
figures/                   Publication figures from the preprint
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository, branch `main`, file `app.py`
5. Deploy

## Citation

Ravikumar, A., Kaya, T., Artan, N.S., Taber, C., Morris, J.R., Raval, M.S. (2026). *Penalty-kill personnel deployment and offensive-value exposure in NCAA ice hockey: a box-score decision-support framework.* SportRxiv. [doi.org/10.51224/SportRxiv.972](https://doi.org/10.51224/SportRxiv.972)
