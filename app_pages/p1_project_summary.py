import streamlit as st


def p1_project_summary_body():
    st.write("### Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"**Powdery mildew** is a common fungal disease affecting a wide range of plant species."
        f" It is caused by various species of ascomycete fungi and thrives "
        f"in environments with moderate temperatures and high humidity. "
        f"In cherry trees, the disease is specifically caused by *Podosphaera clandestina*,"
        f" a parasitic fungus that significantly impacts both leaf and fruit quality.\n\n"

        f"The disease manifests as distinctive white, powder-like spots that typically "
        f"appear on the upper surface of leaves and stems."
        f"  In the early stages, light-green, circular lesions may form on either side of the leaf."
        f"  These lesions can develop into a cotton-like fungal growth,"
        f" especially on newer foliage and, in severe cases, on the fruit itself.\n\n"
        f"This results in:\n"
        f"* Slowed plant growth\n"
        f"* Reduced fruit yield\n"
        f"* Compromised product quality\n"
        )

    st.info(
        f"The client, **Farmy & Foods**, is currently facing a widespread outbreak "
        f"across several of its cherry tree plantations. "
        f"The existing method of disease detection involves manual leaf inspection, "
        f"which is labor-intensive and time-consuming, "
        f"with each tree taking approximately 30 minutes to assess.\n\n"

        f"To address this challenge, the client has commissioned the "
        f"development of a **Machine Learning (ML)** solution capable of "
        f"automatically detecting signs of powdery mildew from leaf images."
        f"The goal is to streamline the inspection process and improve "
        f"early detection, enabling faster treatment and minimizing crop loss."
        )

    st.write(
        f"For additional information, please see the Project "
        f"[README](https://github.com/JedDizon/cherryleafmildewdetector-ci-project5#readme)"
        f" file.")

    st.success(
        f"The project has two **Business Requirements**:\n\n"
        f"1 - Enable the client to visually compare and distinguish between "
        f"healthy and infected cherry leaves using an interactive interface "
        f"or dashboard.\n\n"
        f"2 - Develop a system that can predict if a cherry leaf is healthy "
        f"or infected with powdery mildew based on an image input.\n\n"
        )
    
    st.warning(
        f"**Project Dataset**\n\n"
        f"* The dataset provided contains a total of 4208 images composed of "
        f"healthy and affected leaves.\n"
        f"* The dataset is available for download on "
        f"[Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves)")

