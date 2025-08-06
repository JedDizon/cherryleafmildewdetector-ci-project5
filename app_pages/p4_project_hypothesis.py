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
        f"Model performance shows the model predicting with a 99.76% degree of"
        f"accuracy."
    )

    st.write("---")

    st.info(
        f"**Hypothesis 2**\n\n"
        f"Resizing images to a smaller fixed resolution (e.g., 50x50) "
        f"retains enough visual detail for accurate classification."
    )

    st.success(
        f"**Validation 2**\n\n"
        f"Using smaller image sizes reduces computational cost while preserving "
        f"key features. The original dataset images are 256x256 pixels, but "
        f"training with this resolution typically results in a model larger "
        f"than 100MB—exceeding GitHub’s standard file size limit.\n\n"
        f"Git LFS was avoided by using smaller input dimensions (160x160). "
        f"This approach maintains performance while keeping the model under "
        f"the 100MB threshold."
    )

    st.write("---")

    st.info(
        f"**Hypothesis 3**\n\n"
        f"Average and variability images for each class (healthy vs. infected)"
        f" will show distinct patterns."
    )

    st.success(
        f"**Validation 3**\n\n"
        f"Model performance shows differences with can be observed"
        f"in the form of visible white spot/streak patterns.\n"
        f"Average images also reveal a color difference where infected leaves "
        f"appear as a lighter shade of green compared to healthy ones."
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
        f"prediction of the precense of mildew on a cherry leaf."
    )
