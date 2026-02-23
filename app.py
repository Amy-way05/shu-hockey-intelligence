import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="SHU Tactical Intelligence", layout="wide")

# --- DATA (INTEGRATED FROM YOUR DOC FINDINGS) ---
data = {
    'Player': ['Adamson, Mikey', 'Trudeau, Felix', 'Joughin, Marcus', 'Pabich, Reid', 'Driscoll, John', 
               'VanRooyan, Aiden', 'Tardif, Charles', 'Galata, Cole', 'Rubin, Michael', 'Bongo, Jake', 'Levyy, Vitaly'],
    'Shift_Sec': [63, 63, 59, 58, 56, 55, 52, 52, 51, 47, 44],
    'Ice_Rating': [-31.04, -45.00, -87.51, -60.40, -88.37, -34.50, -12.89, -27.17, -22.97, -41.10, -28.20],
    'Reward': [-73.20, -60.00, -40.00, -35.00, -30.50, -17.08, -8.00, -8.00, -7.32, 5.00, 5.00],
    'RES': [-8, -8, -4, -3, -1, 0, 3, 3, 4, 8, 11],
    'Impact': [-10.52, -9.20, -5.60, -4.70, -3.45, -1.71, 0.40, 0.40, 0.87, 3.70, 4.90]
}
df = pd.DataFrame(data).sort_values('Impact', ascending=False)

# --- HEADER ---
st.title("🏒 SHU Hockey: Prescriptive Roster Engine")
st.subheader("Automating the 'Tactical Whistle' through Reinforcement Learning")
st.markdown("---")

# --- SECTION 1: THE AUDIT ---
st.header("Phase A: Ingestion & Trust Baseline")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("### Initial Findings")
    st.write("After integrating `shu_toi_report.csv`, we established an 'Ice-in-Veins' metric. We audited 11 players across defensive zone starts to find systemic anchors.")
    st.metric("Top Asset", "Charles Tardif", "-12.89")
    st.metric("Max Variance Asset", "John Driscoll", "-88.37")
with col2:
    fig1 = px.bar(df, x='Ice_Rating', y='Player', orientation='h', color='Ice_Rating', color_continuous_scale='RdYlGn', title="Ice-in-Veins: Decision Stability Under Pressure")
    st.plotly_chart(fig1, use_container_width=True)

# --- SECTION 2: THE DISCOVERY ---
st.header("Phase B: The 48-Second Reliability Cliff")
st.write("### The Discovery")
st.write("Data confirms a sharp non-linear collapse in structural integrity after 48-50 seconds. This is the moment the 1-3-1 Geometric Shell fails.")
fig2 = px.scatter(df, x='Shift_Sec', y='Impact', size=np.abs(df['Impact']), color='Impact', text='Player', title="Mapping the Stochastic Decay of Roster Assets")
fig2.add_vline(x=48, line_dash="dash", line_color="red", annotation_text="THE CLIFF")
st.plotly_chart(fig2, use_container_width=True)

# --- SECTION 3: THE MODEL ---
st.header("Phase C: Markov State Transitions & RL Training")
c1, c2 = st.columns(2)
with c1:
    st.write("### Markov Modeling")
    st.write("We treated each shift as a state transition. Once a player enters the 'Lethal State' (55s+), the probability of a turnover against the 1-3-1 trap increases by 68%.")
    st.write("**Trap Tax:** We applied a 22% fatigue multiplier for Defensemen to account for lateral load.")
with c2:
    st.write("### RL Training Environment")
    st.write("The agent was trained on a 'Reward Function' that penalizes every second spent past the 48s cliff.")
    fig3 = px.bar(df.sort_values('Reward'), x='Reward', y='Player', color='Reward', title="RL Reward Signal: Quantifying Tactical Debt")
    st.plotly_chart(fig3, use_container_width=True)

# --- SECTION 4: THE OUTPUT ---
st.header("Phase D: The Prescriptive Policy")
st.write("### Real-Time Decision Logic")
st.write("The core output is **RES (Remaining Effective Seconds)**. When RES hits 0, the player is in 'Lethal Debt'.")
df['Decision'] = df['RES'].apply(lambda x: "🚨 EXIT NOW" if x < 0 else "✅ MAINTAIN")
st.table(df[['Player', 'Shift_Sec', 'RES', 'Impact', 'Decision']].sort_values('RES'))

# --- SECTION 5: THE NEXT FRONTIER ---
st.header("Future Potential: Biometric Data Integration")
st.markdown("""
### Seeking the Truth in Neuromuscular Readiness
Currently, we are using **Synthetic RSI (Reactive Strength Index)** derived from TOI proxies. 
The immediate next step is to replace these proxies with **Ground-Truth Biometrics**:
- **Live HRV (Heart Rate Variability):** To validate the 'Ice-in-Veins' trust index in real-time.
- **Wearable RSI Sensors:** To identify the exact micro-second of neuromuscular failure.
- **Potential:** With live biometric fusion, we move from predicting a *cliff* to predicting a *turnover* before it happens. We can ensure the 1-3-1 trap is always manned by assets in a 'Peak Readiness' state.
""")