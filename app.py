import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import os

# ── CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Penalty-Kill Decision Support Framework",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "data/"
    return {
        'policy': pd.read_csv(base + 'optimal_policy.csv'),
        'qcurve': pd.read_csv(base + 'qlearner_convergence.csv'),
        'rewards': pd.read_csv(base + 'reward_table.csv'),
        'loso': pd.read_csv(base + 'loso_stability.csv'),
        'bootstrap': pd.read_csv(base + 'bootstrap_game_level.csv'),
        'coefs': pd.read_csv(base + 'coefficient_cis.csv'),
        'tost': pd.read_csv(base + 'tost_bootstrap.csv'),
        'pareto': pd.read_csv(base + 'pareto_stability.csv'),
        'calibration': pd.read_csv(base + 'cross_program_calibration.csv'),
        'summary': pd.read_csv(base + 'intelligence_summary.csv'),
        'decisions': pd.read_csv(base + 'tactical_decision_matrix.csv'),
    }

data = load_data()

# ── STYLING ─────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2rem !important; }
    .metric-card {
        background: #f8f7f4;
        border: 1px solid #e5e2dc;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #1a5276; }
    .metric-lab { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .citation { font-size: 0.78rem; color: #888; border-top: 1px solid #e5e2dc; padding-top: 12px; margin-top: 24px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Section", [
        "Overview",
        "Q-Learning Model",
        "Markov Transition Rewards",
        "Optimal Policy Explorer",
        "Model Validation",
        "Cross-Programme Calibration",
        "Publication Figures",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Publication**")
    st.markdown("[SportRxiv Preprint →](https://doi.org/10.51224/SportRxiv.972)")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#999;'>
    All player names, programme names, and conference affiliations
    have been anonymized. Tactical outputs are presented at the
    aggregate level to preserve competitive confidentiality.
    </div>
    """, unsafe_allow_html=True)

# ── OVERVIEW ────────────────────────────────────────────────
if page == "Overview":
    st.title("Penalty-Kill Personnel Deployment and Offensive-Value Exposure")
    st.markdown("**A Box-Score Decision-Support Framework for NCAA Division I Ice Hockey**")
    st.markdown("---")

    st.markdown("""
    This dashboard presents the analytical outputs of a first-author published research
    study examining how penalty-kill deployment decisions affect goal-against risk in
    NCAA Division I ice hockey. The work integrates reinforcement learning (Q-learning),
    Markov Chain transition modeling, and rigorous cross-validation to produce an
    actionable decision-support framework for coaching staff.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">13,777</div><div class="metric-lab">Penalty Outcomes Analyzed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">55</div><div class="metric-lab">NCAA D1 Programmes</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">8</div><div class="metric-lab">Game States Modeled</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">10,000</div><div class="metric-lab">Q-Learning Iterations</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Research Design")
    st.markdown("""
    The study addresses a core coaching problem: when a team is shorthanded due to a
    penalty, which personnel deployment strategy minimizes the probability of
    conceding a goal, given the current game state (period, score differential, and
    pressure level)?

    Two complementary modeling approaches were applied to the same dataset:

    **Q-Learning (Reinforcement Learning):** An 8-state reward structure trained over
    10,000 iterations to learn which deployment actions correlate with reduced scoring
    risk across different game contexts.

    **Markov Chain Transition Maps:** State-transition probabilities capturing how
    penalty situations evolve from one game state to another, providing interpretable
    probability estimates a coaching staff can read directly.

    **Validation:** Leave-one-season-out (LOSO) cross-validation across all available
    seasons, supplemented by 500-game bootstrap resampling and equivalence testing
    (TOST) to confirm that observed patterns hold beyond the training data.
    """)

    smry = data['summary']
    st.markdown("### Key Intelligence Metrics")
    for _, row in smry.iterrows():
        st.markdown(f"**{row['Metric']}:** {row['Value']}")

    st.markdown('<div class="citation">Ravikumar, A., Kaya, T., Artan, N.S., Taber, C., Morris, J.R., Raval, M.S. (2026). <i>Penalty-kill personnel deployment and offensive-value exposure in NCAA ice hockey: a box-score decision-support framework.</i> SportRxiv. <a href="https://doi.org/10.51224/SportRxiv.972">doi.org/10.51224/SportRxiv.972</a></div>', unsafe_allow_html=True)


# ── Q-LEARNING ──────────────────────────────────────────────
elif page == "Q-Learning Model":
    st.title("Q-Learning Convergence Analysis")
    st.markdown("---")

    qc = data['qcurve']
    observed = qc[qc['type'] == 'observed']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=observed['n'], y=observed['agreement'],
        mode='lines+markers', name='Policy Agreement',
        line=dict(color='#1a5276', width=2.5),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=observed['n'],
        y=observed['agreement'] + observed['sd'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=observed['n'],
        y=observed['agreement'] - observed['sd'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(26,82,118,0.15)', name='±1 SD'
    ))
    fig.add_vline(x=521, line_dash="dash", line_color="#c0392b",
                  annotation_text="Study dataset (n=521)")
    fig.update_layout(
        title="Q-Learner Policy Agreement vs. Simulated Event Count",
        xaxis_title="Number of Events (log scale)",
        yaxis_title="Policy Agreement Rate",
        xaxis_type="log",
        yaxis=dict(range=[0.75, 1.05]),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The Q-learner achieves a policy agreement rate of approximately 0.94 to 0.96
    across a wide range of simulated event counts (100 to 2,000), with the study
    dataset sitting at n=521 within the flat convergence region. Agreement
    deteriorates beyond approximately 5,000 events, consistent with increased
    state-space sparsity at higher granularity.
    """)


# ── MARKOV REWARDS ──────────────────────────────────────────
elif page == "Markov Transition Rewards":
    st.title("State-Conditioned Reward Structure")
    st.markdown("---")

    rewards = data['rewards'].sort_values('reward', ascending=False)

    fig = px.bar(
        rewards, x='reward', y='state', orientation='h',
        color='reward', color_continuous_scale='RdYlGn',
        title="Penalty-Kill Survival Reward by Game State"
    )
    fig.update_layout(
        yaxis_title="Game State",
        xaxis_title="Reward (Survival Probability)",
        template="plotly_white",
        height=max(400, len(rewards)*25)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### State Frequency and Goal-Against Rate")
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(rewards.sort_values('n', ascending=False).head(15),
                      x='state', y='n', title="Sample Size by State",
                      color_discrete_sequence=['#1a5276'])
        fig2.update_layout(template="plotly_white", xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.scatter(rewards, x='pga', y='reward', size='n',
                          hover_data=['state'],
                          title="Predicted Goal-Against Rate vs. Reward",
                          color_discrete_sequence=['#c0392b'])
        fig3.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig3, use_container_width=True)


# ── OPTIMAL POLICY ──────────────────────────────────────────
elif page == "Optimal Policy Explorer":
    st.title("Optimal Deployment Policy by Game State")
    st.markdown("---")

    policy = data['policy']

    st.markdown("### Filter by Game Context")
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Period", ['All'] + sorted(policy['Period'].unique().tolist()))
    with col2:
        score = st.selectbox("Score State", ['All'] + sorted(policy['Score'].unique().tolist()))
    with col3:
        pressure = st.selectbox("Pressure Level", ['All'] + sorted(policy['Pressure'].unique().tolist()))

    filtered = policy.copy()
    if period != 'All':
        filtered = filtered[filtered['Period'] == period]
    if score != 'All':
        filtered = filtered[filtered['Score'] == score]
    if pressure != 'All':
        filtered = filtered[filtered['Pressure'] == pressure]

    st.markdown(f"**Showing {len(filtered)} state configurations**")

    action_colors = {
        'std_PK': '#1a5276', 'agg_rot': '#c0392b', 'imm_sub': '#27ae60',
        'keep_rested': '#f39c12', 'short_shifts': '#8e44ad'
    }

    fig = px.scatter(
        filtered, x='Expected_Reward', y='Q_Gap',
        color='Optimal_Action', size='Training_Samples',
        hover_data=['Period', 'Score', 'Pressure'],
        title="Policy Landscape: Expected Reward vs. Decision Confidence",
        color_discrete_map=action_colors
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Policy Table")
    display_cols = ['Period', 'Score', 'Pressure', 'Optimal_Action', 'Expected_Reward', 'Q_Gap', 'Training_Samples']
    st.dataframe(
        filtered[display_cols].sort_values('Expected_Reward', ascending=False),
        use_container_width=True,
        height=400
    )

    st.markdown("""
    **Action definitions:**
    Standard PK deployment (std_PK), aggressive rotation (agg_rot),
    immediate substitution (imm_sub), keep rested personnel (keep_rested),
    shortened shift cycles (short_shifts). Q_Gap measures the margin between
    the best and second-best action, indicating decision confidence.
    """)


# ── VALIDATION ──────────────────────────────────────────────
elif page == "Model Validation":
    st.title("Validation: LOSO Cross-Validation and Bootstrap Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["LOSO Stability", "Bootstrap Confidence", "Coefficient Intervals"])

    with tab1:
        loso = data['loso']
        fig = px.bar(loso, x='fold', y='cal_err', color='config',
                     barmode='group', title="Calibration Error by Season Fold",
                     color_discrete_sequence=['#1a5276', '#c0392b', '#27ae60', '#f39c12'])
        fig.update_layout(template="plotly_white", height=400,
                          yaxis_title="Calibration Error", xaxis_title="Hold-Out Season")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Leave-one-season-out cross-validation holds out each season in turn and
        trains the model on the remaining data. Calibration error measures the
        gap between predicted and observed goal-against rates. Values below 0.05
        indicate strong calibration.
        """)

    with tab2:
        boot = data['bootstrap']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=boot['state'], y=boot['obs_ga'], name='Observed',
            marker_color='#1a5276'
        ))
        fig.add_trace(go.Scatter(
            x=boot['state'], y=boot['boot_mean'], mode='markers',
            error_y=dict(type='data',
                         symmetric=False,
                         array=(boot['ci_hi'] - boot['boot_mean']).tolist(),
                         arrayminus=(boot['boot_mean'] - boot['ci_lo']).tolist()),
            name='Bootstrap Mean (95% CI)',
            marker=dict(color='#c0392b', size=10)
        ))
        fig.update_layout(
            title="Observed vs. Bootstrap Goal-Against Rate by State",
            template="plotly_white", height=450,
            xaxis_title="Game State", yaxis_title="Goal-Against Probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        coefs = data['coefs']
        fig = go.Figure()
        for _, row in coefs.iterrows():
            color = '#c0392b' if row['significant'] == 'YES' else '#888'
            fig.add_trace(go.Scatter(
                x=[row['ci_lo'], row['estimate'], row['ci_hi']],
                y=[row['coefficient']]*3,
                mode='lines+markers',
                marker=dict(size=[6, 12, 6], color=color),
                line=dict(color=color, width=2),
                name=row['coefficient'],
                showlegend=False
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="#999")
        fig.update_layout(
            title="Coefficient Estimates with 95% Confidence Intervals",
            xaxis_title="Estimate (log-odds)",
            template="plotly_white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Red markers indicate statistically significant coefficients (95% CI excludes zero).")


# ── CROSS-PROGRAMME ─────────────────────────────────────────
elif page == "Cross-Programme Calibration":
    st.title("External Validation Across 55 NCAA Division I Programmes")
    st.markdown("---")

    cal = data['calibration']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(cal, x='cal_error', nbins=20,
                           title="Distribution of Calibration Error Across Programmes",
                           color_discrete_sequence=['#1a5276'])
        fig.update_layout(template="plotly_white", xaxis_title="Calibration Error",
                          yaxis_title="Number of Programmes", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(cal, x='late_coef', nbins=20,
                           title="Late-Penalty Log-Odds Coefficient Distribution",
                           color_discrete_sequence=['#c0392b'])
        fig.update_layout(template="plotly_white", xaxis_title="Late-Penalty Coefficient",
                          yaxis_title="Number of Programmes", height=400)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(cal, x='cal_error', y='late_coef', size='n',
                     color='late_sig', hover_data=['team'],
                     title="Calibration Error vs. Late-Penalty Effect by Programme",
                     color_discrete_map={True: '#c0392b', False: '#1a5276'})
    fig.update_layout(template="plotly_white", height=500,
                      xaxis_title="Calibration Error",
                      yaxis_title="Late-Penalty Coefficient")
    st.plotly_chart(fig, use_container_width=True)

    sig_count = cal['late_sig'].sum()
    st.markdown(f"""
    Of the 55 programmes analyzed, **{sig_count}** show a statistically significant
    late-penalty effect. The median calibration error across all programmes is
    **{cal['cal_error'].median():.4f}**, indicating the model generalizes well beyond
    the study programme's own data.
    """)


# ── PUBLICATION FIGURES ─────────────────────────────────────
elif page == "Publication Figures":
    st.title("Publication Figures from the Preprint")
    st.markdown("---")
    st.markdown("These figures are reproduced from the published preprint on SportRxiv. Programme names and player identities have been anonymized throughout.")

    figures = [
        ("Figure 1: Model Calibration Across 8 Game States", "figure1_calibration.png",
         "LOSO cross-validated calibration plot showing predicted versus observed goal-against probability for each of the eight game states. Point size is proportional to bin sample size."),
        ("Figure 2: External Validation (Forest Plot)", "figure2_external_validation.png",
         "Random-effects meta-analysis of the late-penalty risk coefficient across 54 external programmes, with the study programme highlighted. The pooled effect is near zero, with the study programme falling within the overall distribution."),
        ("Figure 3: Pareto Frontier (Risk vs. Survival)", "figure3_pareto_frontier.png",
         "Risk-survival tradeoff for penalty-kill unit configurations. Base PK units (PK1 through PK3) are compared against Pareto-optimal frontier configurations (F1 through F3) that minimize offensive-value exposure."),
        ("Figure 4: Q-Learner Convergence", "figure4_qlearner_curve.png",
         "Policy agreement rate as a function of simulated event count. The study dataset (n=521) falls within the flat convergence region, confirming the Q-learner has sufficient data to produce stable policy recommendations."),
        ("Figure 5: Cross-Programme Penalty Distribution", "figure5_forest_plot.png",
         "Second-period penalty concentration across all Division I programmes for two seasons, comparing the study programme against the league-wide distribution."),
    ]

    for title, fname, caption in figures:
        fpath = f"figures/{fname}"
        if os.path.exists(fpath):
            st.markdown(f"### {title}")
            st.image(fpath, use_container_width=True)
            st.caption(caption)
            st.markdown("---")


# ── FOOTER ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="citation">
<b>Full citation:</b> Ravikumar, A., Kaya, T., Artan, N.S., Taber, C., Morris, J.R., Raval, M.S. (2026).
<i>Penalty-kill personnel deployment and offensive-value exposure in NCAA ice hockey:
a box-score decision-support framework.</i> SportRxiv.
<a href="https://doi.org/10.51224/SportRxiv.972">doi.org/10.51224/SportRxiv.972</a><br><br>
All player names, programme names, and conference affiliations have been anonymized.
No tactical strategy, individual performance data, or identifiable game film
is disclosed in this dashboard.
</div>
""", unsafe_allow_html=True)
