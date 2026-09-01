"""
streamlit_app.py
------------------
STAGE IX: Web Application & Automated Report.

This is the single working web app for the AI Factory Intelligence
Command Center. It ties Stages I-VIII together into one flow:

    1. Inputs      -- upload the selected data modalities
                       (image / sensor CSV / maintenance text / manual PDF)
    2. Prediction   -- run each modality's model/agent, show
       & XAI           prediction + confidence + explainability
    3. Digital Twin -- Stage VII's what-if scenario comparison
       & Decision      + Stage VIII's Approve/Reject/Override form
    4. Report       -- generate a downloadable PDF or DOCX
                       factory incident/decision report
    5. Continuous   -- STAGE X (Grand Challenge): every decision from
       Learning         tab 3 feeds a feedback data store; retrain a
                        "trust policy" candidate model, evaluate it
                        against the current champion, log to MLflow,
                        and promote it if it's better.

Run:
    streamlit run streamlit_app.py

Note: the model/agent calls in ingestion/modality_inputs.py are
clearly-marked placeholders. Stage IX's rubric line is the app + the
report, not re-deriving the models that Stages II-VI already score --
swap those functions for your real trained models/RAG pipeline and
nothing else in this file needs to change.
"""

import os
import json
import pandas as pd
import streamlit as st

from supervisor.human_approval import HumanSupervisor
from mlops.mlflow_tracking import log_decision_run, log_retraining_run
from reporting.report_generator import (
    build_report_context, generate_pdf_report, generate_docx_report, report_filename,
)
from ingestion.modality_inputs import (
    analyze_image, analyze_sensor_csv, analyze_maintenance_text,
    retrieve_manual_evidence, build_xai_summary, build_agent_trace,
)
from feedback.feedback_store import record_feedback, load_feedback_dataset, dataset_summary
from feedback.retrain import (
    train_candidate_model, load_champion, candidate_beats_champion,
    save_candidate_model, promote_to_champion, next_version_name,
)

OUTPUTS_DIR = "outputs"
LOG_PATH = os.path.join(OUTPUTS_DIR, "approval_log.json")
FINAL_DECISION_PATH = os.path.join(OUTPUTS_DIR, "final_decision.json")

CANDIDATE_REPORT_PATHS = [
    os.path.join(OUTPUTS_DIR, "digital_twin_report.json"),
    os.path.join("..", "ai_factory_stage7", "outputs", "digital_twin_report.json"),
    os.path.join("..", "ai_factory_stage7", "ai_factory_stage7", "outputs", "digital_twin_report.json"),
]

st.set_page_config(page_title="AI Factory Command Center", page_icon="🏭", layout="wide")


# ----------------------------------------------------------------------
# Load the Stage VII digital-twin report (auto-detect, manual path, or upload)
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


# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
for key, default in {
    "vision_result": None,
    "sensor_result": None,
    "text_result": None,
    "rag_evidence": [],
    "decision_record": None,
}.items():
    st.session_state.setdefault(key, default)


st.title("🏭 AI Factory Intelligence Command Center")
st.caption(
    "Autonomous Manufacturing Intelligence & Digital Twin -- end-to-end demo: "
    "upload data -> see predictions & explainability -> review the digital twin's "
    "recommendation -> approve/reject/override -> download the report."
)

report, report_path = load_report()
if report is None:
    st.stop()

machine_id = report.get("machine_id", "UNKNOWN")
recommendation = report.get("recommendation", "UNKNOWN")
rationale = report.get("rationale", "")
scenarios = report.get("scenarios", [])
scenario_names = [s["scenario"] for s in scenarios]

tab_inputs, tab_predict, tab_twin, tab_report, tab_learn = st.tabs(
    ["📥 1. Data Inputs", "🔎 2. Prediction & XAI", "🌐 3. Digital Twin & Decision",
     "📄 4. Report", "🔁 5. Continuous Learning"]
)

# ========================================================================
# TAB 1 -- Data inputs (multimodal)
# ========================================================================
with tab_inputs:
    st.subheader(f"Machine: {machine_id}")
    st.caption(f"Digital-twin report loaded from: `{report_path}`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🖼️ Product / component image**")
        img_file = st.file_uploader("Upload a product or component photo", type=["png", "jpg", "jpeg"], key="img")
        if img_file is not None:
            st.image(img_file, width=240)
            st.session_state.vision_result = analyze_image(img_file.getvalue(), img_file.name)
            st.success(f"Vision agent: {st.session_state.vision_result['value']} "
                       f"({st.session_state.vision_result['confidence']:.0%} confidence)")

        st.markdown("**📝 Maintenance / incident note**")
        note_text = st.text_area("Paste a maintenance note or incident description", height=100, key="note")
        if note_text:
            st.session_state.text_result = analyze_maintenance_text(note_text)
            st.info(f"NLP agent: {st.session_state.text_result['value']}")

    with c2:
        st.markdown("**📈 Sensor readings (time series)**")
        csv_file = st.file_uploader("Upload a sensor CSV (temperature/vibration/pressure/RPM...)", type=["csv"], key="csv")
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(), use_container_width=True, hide_index=True)
            st.session_state.sensor_result = analyze_sensor_csv(df)
            st.success(f"Predictive maintenance agent: {st.session_state.sensor_result['value']}")

        st.markdown("**📄 Machine manual / SOP (for RAG)**")
        pdf_file = st.file_uploader("Upload a manual or SOP PDF", type=["pdf"], key="pdf")
        query = st.text_input("What should the knowledge agent look up?", value=rationale[:80] if rationale else "")
        if st.button("Retrieve evidence") and query:
            st.session_state.rag_evidence = retrieve_manual_evidence(
                query, pdf_file.name if pdf_file else None
            )
            for ev in st.session_state.rag_evidence:
                st.write(f"**Source:** {ev['source']}")
                st.caption(ev["excerpt"])

    st.info(
        "All uploads here are optional -- the digital-twin recommendation on the "
        "next tabs works from the Stage VII report regardless. Uploading data lets "
        "the Prediction & XAI and Report tabs show real per-modality evidence."
    )

# ========================================================================
# TAB 2 -- Predictions & explainability
# ========================================================================
predictions = {}
if st.session_state.vision_result:
    predictions["Vision (defect check)"] = st.session_state.vision_result
if st.session_state.sensor_result:
    predictions["Predictive maintenance (sensor trend)"] = st.session_state.sensor_result
if st.session_state.text_result:
    predictions["NLP (maintenance note urgency)"] = st.session_state.text_result

xai = build_xai_summary(predictions) if predictions else {}

with tab_predict:
    st.subheader("Model predictions & confidence")
    if not predictions:
        st.info("Upload data on the **Data Inputs** tab to see per-modality predictions here.")
    else:
        for name, result in predictions.items():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{name}**")
                st.write(result["value"])
                if result.get("note"):
                    st.caption(result["note"])
            with col_b:
                st.metric("Confidence", f"{result['confidence']:.0%}")

        st.divider()
        st.subheader("Explainable AI")
        st.write(f"**Method:** {xai.get('method', 'N/A')}")
        for feat in xai.get("top_features", []):
            st.write(f"- {feat}")
        st.caption(xai.get("narrative", ""))

# ========================================================================
# TAB 3 -- Digital twin comparison + human decision (Stage VII + VIII)
# ========================================================================
with tab_twin:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("AI Recommendation", recommendation)
    with col2:
        st.info(rationale)

    st.subheader("Scenario comparison")
    df_sc = pd.DataFrame(scenarios)
    if not df_sc.empty:
        df_display = df_sc.rename(columns={
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
    st.subheader("Your decision")

    decision_choice = st.radio(
        "What do you want to do?",
        options=["Approve", "Reject", "Override"],
        horizontal=True,
    )

    override_target = None
    if decision_choice == "Override":
        override_target = st.selectbox("Choose the scenario to execute instead:", scenario_names)

    reviewer_name = st.text_input("Your name", key="reviewer_name")
    reviewer_comment = st.text_area("Comment / reason (optional)", key="reviewer_comment")

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
            source_report=report_path,
        )

        final_decision = record.to_dict()
        final_decision["source_report"] = report_path
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(FINAL_DECISION_PATH, "w") as f:
            json.dump(final_decision, f, indent=2, default=str)

        run_id = log_decision_run(report, decision, final_action)
        st.session_state.decision_record = final_decision

        # STAGE X: feed this decision into the continuous-learning feedback
        # store (AI Prediction -> Human Decision -> Feedback -> Data Store).
        record_feedback(report, final_decision)

        st.success(f"Decision recorded: **{decision}** -> final action: **{final_action}**")
        st.caption(f"MLflow run id: `{run_id}`")
        st.caption("Also added to the continuous-learning feedback dataset (see tab 5).")
        st.json(final_decision)

    st.divider()
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

# ========================================================================
# TAB 4 -- Downloadable report
# ========================================================================
with tab_report:
    st.subheader("Generate the factory incident / decision report")

    decision_record = st.session_state.decision_record
    if decision_record is None and os.path.exists(FINAL_DECISION_PATH):
        with open(FINAL_DECISION_PATH) as f:
            decision_record = json.load(f)

    if decision_record is None:
        st.info(
            "Submit a decision on the **Digital Twin & Decision** tab first -- "
            "the report includes the human supervisor's final call."
        )
    else:
        agent_trace = build_agent_trace(
            st.session_state.vision_result,
            st.session_state.sensor_result,
            st.session_state.text_result,
            st.session_state.rag_evidence,
            recommendation,
        )
        ctx = build_report_context(
            report=report,
            decision_record=decision_record,
            predictions=predictions,
            xai=xai,
            rag_evidence=st.session_state.rag_evidence,
            agent_trace=agent_trace,
        )

        st.write(f"Report will cover machine **{ctx['machine_id']}**, "
                 f"decision **{ctx['human_decision']}** -> **{ctx['final_action']}**, "
                 f"with {len(predictions)} model prediction(s) and "
                 f"{len(agent_trace)} agent step(s) attached.")

        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = generate_pdf_report(ctx)
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=report_filename(ctx, "pdf"),
                mime="application/pdf",
                type="primary",
            )
        with col2:
            docx_bytes = generate_docx_report(ctx)
            st.download_button(
                "⬇️ Download DOCX report",
                data=docx_bytes,
                file_name=report_filename(ctx, "docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

# ========================================================================
# TAB 5 -- STAGE X (Grand Challenge): Continuous Learning loop
#   AI Prediction -> Human Decision -> Feedback -> Data Store ->
#   Retraining -> Evaluation -> MLflow -> Candidate Model
# ========================================================================
with tab_learn:
    st.subheader("🔁 Continuous learning loop")
    st.caption(
        "Every Approve / Reject / Override decision on tab 3 is written to a "
        "feedback dataset here. This trains a small, transparent 'trust policy' "
        "model that predicts whether a human is likely to approve a given AI "
        "recommendation -- and only promotes a new candidate if it beats the "
        "current champion on held-out data."
    )

    summary = dataset_summary()
    s1, s2, s3 = st.columns(3)
    s1.metric("Feedback rows collected", summary["rows"])
    s2.metric("Approved examples", summary["approved"])
    s3.metric("Not-approved examples", summary["not_approved"])

    feedback_df = load_feedback_dataset()
    if not feedback_df.empty:
        with st.expander("View feedback dataset"):
            st.dataframe(feedback_df, use_container_width=True, hide_index=True)
    else:
        st.info("No feedback yet -- submit a decision on tab 3 first.")

    st.divider()

    champion = load_champion()
    st.subheader("🏆 Current champion model")
    if champion:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.metric("Version", champion["version"])
            st.metric("F1 score", champion["metrics"].get("f1", "N/A"))
        with cc2:
            st.json(champion["metrics"])
        st.caption(f"Promoted at: {champion.get('promoted_at_utc', 'unknown')}")
    else:
        st.info("No champion model has been promoted yet.")

    st.divider()
    st.subheader("🧪 Train a new candidate")

    if not summary["ready_to_train"]:
        st.warning(
            "Need at least 2 APPROVED and 2 not-approved (rejected/overridden) "
            "decisions before a meaningful model can be trained. Keep using tab 3."
        )

    if st.button("Retrain candidate model", disabled=not summary["ready_to_train"]):
        model, metrics = train_candidate_model()

        if model is None:
            st.error(metrics.get("note", "Could not train a candidate model."))
        else:
            st.success("Candidate trained.")
            st.json(metrics)

            is_better, reason = candidate_beats_champion(metrics)
            st.write(f"**Comparison vs. champion:** {reason}")

            version_name = next_version_name()
            model_path = save_candidate_model(model, version_name)

            run_id = log_retraining_run(
                params={"model_version": version_name, "train_rows": metrics.get("train_rows"),
                        "test_rows": metrics.get("test_rows")},
                metrics={k: v for k, v in metrics.items() if k in ("accuracy", "precision", "recall", "f1")},
                artifact_paths={"candidate_model": model_path},
                run_name=f"retrain_{version_name}",
            )
            st.caption(f"MLflow run id: `{run_id}`")

            st.session_state["last_candidate"] = {
                "version_name": version_name,
                "model_path": model_path,
                "metrics": metrics,
                "is_better": is_better,
            }

    last_candidate = st.session_state.get("last_candidate")
    if last_candidate:
        st.divider()
        st.subheader("✅ Promote candidate to production")
        st.write(
            f"Candidate **{last_candidate['version_name']}** "
            f"(F1={last_candidate['metrics'].get('f1', 'N/A')}) -- "
            f"{'beats' if last_candidate['is_better'] else 'does NOT beat'} the current champion."
        )
        promote_notes = st.text_input(
            "Promotion notes (optional)",
            value="Promoted after continuous-learning retrain cycle.",
            key="promote_notes",
        )
        if st.button("Promote to champion", disabled=not last_candidate["is_better"]):
            meta = promote_to_champion(
                last_candidate["version_name"],
                last_candidate["model_path"],
                last_candidate["metrics"],
                notes=promote_notes,
            )
            st.success(f"Promoted {meta['version']} to champion.")
            st.session_state.pop("last_candidate", None)
            st.rerun()
