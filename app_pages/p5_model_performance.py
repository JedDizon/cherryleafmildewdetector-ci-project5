import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def p5_model_performance_body():
    """
    Display model performance Metrics
    text and images
    """
    version = 'v2'

    st.write("### Train, Validation and Test Set Frequencies")

    distribute_label = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(distribute_label,
             caption='Distribution of Train, Validation and Test Sets')

    st.info(
        f"### Dataset Splitting Overview\n"
        f"* **The dataset was divided into three subsets**:"
        f" training, validation, and test."
        f"This is a common strategy in machine learning for "
        f"building and evaluating models.\n\n"
        f"* **Training set**: The largest portion of the data, "
        f"used to teach the model to distinguish between the image classes. "
        f"A larger training set helps the model learn more effectively.\n\n"
        f"* **Validation set**: Used during model development to fine-tune "
        f"parameters and improve performance without biasing the model to "
        f"the test data.\n\n"
        f"* **Test set**: Held back until the very end to provide an unbiased "
        f"evaluation of the model’s ability to generalize to new, unseen data."
    )

    distribute_label = plt.imread(
        f"outputs/{version}/sets_distribution_pie.png")
    st.image(
        distribute_label,
        caption="Pie chart of Distribution of Train, Validation and Test Sets"
    )

    st.write("---")

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.info(
        f"* The graphs above indicate that the model achieved "
        f"a consistently high level of accuracy.\n"
        f"* Performance on the training set improved rapidly "
        f"during the initial epochs. "
        f"Although there was a brief dip in accuracy at epoch 4, "
        f"the model quickly recovered in the following "
        f"epoch and continued to improve steadily.\n"
        f"* The loss graph, which measures how far the model’s predictions "
        f"deviate from the actual values, also reflects strong performance"
        f" on both the training and validation sets."
    )

    st.write("---")

    st.write("### Machine Learning Pipeline Overview")
    st.warning(
        f"The machine learning pipeline for this project followed "
        f"the below steps:\n\n"
        f"**1. Data Preprocessing**: Images were preprocessed to ensure "
        f"consistent input size, enhanced for clarity, and normalized to "
        f"standardize pixel values.\n\n"
        f"**2. Model Training**: A CNN classified cherry leaves as "
        f"'healthy' or 'infected'.\n\n"
        f"**3. Model Evaluation**: The model was evaluated on the test set for"
        f" an unbiased assessment of its ability to analyze new data.\n\n"
        f"**4. Optimization**: The model was iteratively refined, achieving "
        f"99% accuracy on the test set.\n\n"
        f"This pipeline allows for reproducibility and scalability, "
        f"ensuring the model can be used for future predictions "
        f"or improvements."
    )

    st.write("---")

    st.write("### Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version),
                 index=['Loss', 'Accuracy']))
    st.info(
        f"* It was requested by the client at the beginning of this project "
        f"for an ML model with a performance criteria that could predict"
        f" with at least 97% accuracy if a leaf was healthy "
        f"or infected with powdery mildew.\n"
        f"* In the above table, the model predicts "
        f"with a **99%** accuracy on the status of"
        f" images in the test dataset. "
    )

    st.write("### Project Conclusions")

    st.success(
        f"**Business Requirement 1**: conduct a study to visually compare "
        f"and distinguish between healthy and infected cherry leaves.\n\n"
        f"* Requirement satisfied.\n"
        f"* See Page 2 (Visual Studies). Average image of healthy vs infected "
        f"leaves. Main differences are visible white spot/streak patterns "
        f"on infected leaves & infected leaves appearing as a lighter "
        f"shade of green compared to healthy ones."
    )

    st.success(
        f"**Business Requirement 2**: predicting if a cherry leaf is healthy "
        f"or infected with powdery mildew based on an image input.\n\n"
        f"* Requirement satisfied.\n"
        f"* See Page 3 (Mildew Detector). ML Model allows users can upload "
        f"cherry leaf  image(s) and receive a prediction with 99% "
        f"accuracy if it is healthy or infected with powdery mildew."
    )
