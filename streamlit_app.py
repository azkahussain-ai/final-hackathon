"""
streamlit_app.py
------------------
STAGE VIII: Human-in-the-Loop Approval -- Streamlit Web UI

A browser-based version of demo.py. Loads the Digital Twin report from
Stage VII, shows the AI's recommendation and full scenario comparison,
and lets a human supervisor Approve / Reject / Override it through a
web form. Every decision is written to the same JSON audit log used by
the CLI version, so both can be used interchangeably.

Run:
    streamlit run streamlit_app.py
"""

import os
import json
import pandas as pd
import streamlit as st

from supervisor.human_approval import HumanSupervisor

OUTPUTS_DIR = "outputs"
LOG_PATH = os.path.join(OUTPUTS_DIR, "approval_log.json")
FINAL_DECISION_PATH = os.path.join(OUTPUTS_DIR, "final_decision.json")

CANDIDATE_REPORT_PATHS = [
    os.path.join(OUTPUTS_DIR, "digital_twin_report.json"),
    os.path.join("..", "ai_factory_stage7", "outputs", "digital_twin_report.json"),
    os.path.join("..", "ai_factory_stage7", "ai_factory_stage7", "outputs", "digital_twin_report.json"),
]

st.set_page_config(page_title="Stage VIII -- Human Approval", page_icon="🏭", layout="wide")


# ----------------------------------------------------------------------
# Load the Stage VII report (auto-detect, manual path, or file upload)
# ----------------------------------------------------------------------
def load_report():
    for path in CANDIDATE_REPORT_PATHS:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f), path

    st.warning("Could not auto-find `digital_twin_report.json` from Stage VII.")
    col1, col2 = st.columns(2)
    with col1:
        typed_path = st.text_input("Paste the file path instead:")
        if typed_path and os.path.exists(typed_path):
            with open(typed_path, "r") as f:
                return json.load(f), typed_path
    with col2:
        uploaded = st.file_uploader("...or upload digital_twin_report.json", type="json")
        if uploaded is not None:
            return json.load(uploaded), "(uploaded file)"

    return None, None


st.title("🏭 Stage VIII -- Human Supervisor Approval")
st.caption(
    "The Digital Twin (Stage VII) can only recommend an action. "
    "Nothing runs on the real machine until a human reviews and approves it here."
)

report, report_path = load_report()
if report is None:
    st.stop()

st.success(f"Loaded report from: `{report_path}`")

machine_id = report.get("machine_id", "UNKNOWN")
recommendation = report.get("recommendation", "UNKNOWN")
rationale = report.get("rationale", "")
scenarios = report.get("scenarios", [])
scenario_names = [s["scenario"] for s in scenarios]

# ----------------------------------------------------------------------
# AI recommendation summary
# ----------------------------------------------------------------------
st.header(f"Machine: {machine_id}")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("AI Recommendation", recommendation)
with col2:
    st.info(rationale)

st.subheader("Scenario comparison")
df = pd.DataFrame(scenarios)
if not df.empty:
    df_display = df.rename(columns={
        "scenario": "Scenario",
        "expected_cost_usd": "Expected cost ($)",
        "p90_cost_usd": "P90 / worst-case cost ($)",
        "unplanned_failure_probability": "Unplanned failure prob.",
        "expected_production_units": "Expected production (units)",
        "expected_downtime_hours": "Expected downtime (h)",
    })

    def highlight_pick(row):
        is_pick = row["Scenario"] == recommendation
        return ["background-color: #d4f7d4" if is_pick else "" for _ in row]

    st.dataframe(
        df_display.style.apply(highlight_pick, axis=1).format({
            "Expected cost ($)": "${:,.0f}",
            "P90 / worst-case cost ($)": "${:,.0f}",
            "Unplanned failure prob.": "{:.1%}",
            "Expected production (units)": "{:,.0f}",
            "Expected downtime (h)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

chart_path = report.get("chart")
if chart_path and os.path.exists(chart_path):
    st.image(chart_path, caption="Scenario cost comparison (from Stage VII)")

st.divider()

# ----------------------------------------------------------------------
# Decision form
# ----------------------------------------------------------------------
st.subheader("Your decision")

decision_choice = st.radio(
    "What do you want to do?",
    options=["Approve", "Reject", "Override"],
    horizontal=True,
)

override_target = None
if decision_choice == "Override":
    override_target = st.selectbox("Choose the scenario to execute instead:", scenario_names)

reviewer_name = st.text_input("Your name")
reviewer_comment = st.text_area("Comment / reason (optional)")

if st.button("Submit decision", type="primary"):
    if decision_choice == "Approve":
        decision, final_action = "APPROVED", recommendation
    elif decision_choice == "Reject":
        decision, final_action = "REJECTED", "NONE"
    else:
        decision, final_action = "OVERRIDDEN", override_target

    supervisor = HumanSupervisor(log_path=LOG_PATH)
    record = supervisor.record_decision(
        report,
        decision=decision,
        final_action=final_action,
        reviewer_name=reviewer_name or "unspecified",
        reviewer_comment=reviewer_comment,
    )

    final_decision = {
        "machine_id": record.machine_id,
        "decision": record.human_decision,
        "final_action": record.final_action,
        "ai_recommendation": record.ai_recommendation,
        "reviewer_name": record.reviewer_name,
        "reviewer_comment": record.reviewer_comment,
        "timestamp_utc": record.timestamp_utc,
        "source_report": report_path,
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(FINAL_DECISION_PATH, "w") as f:
        json.dump(final_decision, f, indent=2, default=str)

    st.success(f"Decision recorded: **{decision}** -> final action: **{final_action}**")
    st.json(final_decision)

st.divider()

# ----------------------------------------------------------------------
# Audit trail
# ----------------------------------------------------------------------
st.subheader("📜 Approval history (audit log)")
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        history = json.load(f)
    if history:
        hist_df = pd.DataFrame(history).sort_values("timestamp_utc", ascending=False)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No decisions recorded yet.")
else:
    st.caption("No decisions recorded yet.")
