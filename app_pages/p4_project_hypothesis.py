import streamlit as st


def p4_project_hypothesis_body():
    st.write("### Project Hypothesis")

    st.info(
        f"**Hypothesis 1**\n\n"
        f"A deep learning model can distinguish between healthy and "
        f"infected cherry leaves with at least 90% accuracy."
    )

    st.success(
        f"**Validation 1**\n\n"
        f"The **Performance on Test Set** table on Model Performance page show "
        f"the model predicting with a 99.76% degree of accuracy. "
        f"This hypothesis can be considered validated."
    )

    st.write("---")

    st.info(
        f"**Hypothesis 2**\n\n"
        f"Resizing images to a smaller fixed resolution (e.g., 50x50) "
        f"retains enough visual detail for accurate classification."
    )

    st.success(
        f"**Validation 2**\n\n"
        f"The original dataset images are 256x256 pixels. By using smaller "
        f"input dimensions (160x160), the Git LFS limit was avoided "
        f"while the model was still able to maintain an accuracy of 99.76%."
        f" This hypothesis can be considered validated."
    )

    st.write("---")

    st.info(
        f"**Hypothesis 3**\n\n"
        f"Average and variability images for each class (healthy vs. infected)"
        f" will show distinct patterns."
    )

    st.success(
        f"**Validation 3**\n\n"
        f"The **Average and Variability Image Differences** and **Average "
        f"Healthy & Infected Cherry Leaves Differences** images on the Visual "
        f"Studies page show the differences in the form ofvisible white spot/"
        f"streak patterns. Average images also reveal a color difference "
        f"where infected leaves appear as a lighter shade of green compared "
        f"to healthy ones. This hypothesis can be considered validated."
    )

    st.write("---")

    st.info(
        f"**Hypothesis 4**\n\n"
        f"Using an ML model will reduce manual inspection time by "
        f"over 90% per tree."
    )

    st.success(
        f"**Validation 4**\n\n"
        f"The mildew detector application allows for a near instantaneous "
        f"prediction of the presence of mildew on a cherry leaf vs the "
        f"problem statement of 30 minutes per tree for inspection. "
        f"This hypothesis can be considered validated. "
    )
