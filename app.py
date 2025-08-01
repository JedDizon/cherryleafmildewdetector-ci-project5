import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.p1_project_summary import p1_project_summary_body
from app_pages.p2_visual_studies import p2_visual_studies_body
from app_pages.p3_mildew_detector import p3_mildew_detector_body
from app_pages.p4_project_hypothesis import p4_project_hypothesis_body
from app_pages.p5_model_performance import p5_model_performance_body

app = MultiPage(app_name="Mildew Detection in Cherry Leaves")

# Add your app pages here using .add_page()
app.app_page("Project Summary", p1_project_summary_body)
app.app_page("Visual Studies", p2_visual_studies_body)
app.app_page("Cherry Leaf Mildew Detector", p3_mildew_detector_body)
app.app_page("Project Hypothesis", p4_project_hypothesis_body)
app.app_page("ML Model Performance", p5_model_performance_body)

app.run()