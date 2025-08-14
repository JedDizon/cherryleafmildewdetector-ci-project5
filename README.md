# Detection of Mildew on Cherry leaves

![Responsive image of site](/assets/images/p5_multimockup.png)

## Table of Contents

- [Detection of Mildew on Cherry leaves](#detection-of-mildew-on-cherry-leaves)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset Content](#dataset-content)
  - [Business Requirements](#business-requirements)
  - [Hypotheses and validation](#hypotheses-and-validation)
  - [The rationale to map the business requirements to the Data Visualisations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
    - [Requirement 1](#requirement-1)
    - [Requirement 2](#requirement-2)
  - [ML business case](#ml-business-case)
  - [Dashboard Design](#dashboard-design)
  - [Deployment](#deployment)
  - [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
  - [Other technologies used](#other-technologies-used)
  - [Issues](#issues)
  - [Testing](#testing)
    - [Manual Testing](#manual-testing)
  - [Validation](#validation)
    - [PEP8 Validation](#pep8-validation)
  - [Credits](#credits)
  - [Acknowledgements](#acknowledgements)

## Introduction

Mildew Detector is a data science and machine learning (ML) initiative focused on using predictive analytics. The project supports Farmy & Foods, an agricultural company specializing in the cultivation and harvesting of various food products, which is currently facing an outbreak of powdery mildew in its cherry orchards.
Currently, tree inspections are carried out manually to identify signs of infection which is both time-consuming and labor-intensive. To streamline this process, we propose the development of a machine learning model capable of analyzing leaf images to detect the presence of mildew. This solution aims to significantly reduce the time required for diagnosis, enabling faster and more accurate treatment of affected trees.

The [project](https://mildew-detector-p5-ci-c8b938269593.herokuapp.com/) is deployed via Streamlit.

[Back to top](#table-of-contents)

## Dataset Content

The dataset consists of 4,208 images of individual cherry leaves. The images were collected from Farmy & Foods’ own cherry crop fields and are labeled as Healthy or Infected with powdery mildew.
While powdery mildew is common across various crops, the client is especially concerned about its impact on cherry plantations.The dataset was originally published on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves) and curated for this project.

[Back to top](#table-of-contents)

## Business Requirements

Farmy & Foods is an agricultural company specializing in the cultivation and harvesting of various food products. One of their key offerings is cherry production, which has recently been impacted by powdery mildew, a common fungal disease affecting a wide variety of plants.
Currently, the detection process for powdery mildew is entirely manual. An employee inspects each cherry tree by:
Collecting leaf samples
Visually checking them for signs of infection
Applying a treatment compound if mildew is found
This process takes:
Approximately 30 minutes per tree for inspection

1 minute per tree for treatment if infected
Given the scale of their operations, the process is highly time-consuming and not scalable. The IT department at Farmy & Foods has proposed the development of a machine learning system capable of instantly identifying powdery mildew from an image of a cherry leaf. This solution aims to:

- Reduce inspection time
- Scale across thousands of trees
- Enable faster intervention and treatment
- Lay the groundwork for similar ML-based solutions across other crops

The dataset used in this project consists of real cherry leaf images captured from Farmy & Foods’ plantations. The primary business requirements for this project are:

- Visual Analysis: Enable the client to visually compare and distinguish between healthy and infected cherry leaves using an interactive interface or dashboard.
- ML-Based Prediction: Develop a system that can predict if a cherry leaf is healthy or infected with powdery mildew based on an image input.

[Back to top](#table-of-contents)

## Hypotheses and validation

| Hypothesis                 |          Validation                       |
|------------------------|---------------------------------------|
| A deep learning model can distinguish between healthy and infected cherry leaves with at least 90% accuracy.|          Model performance shows the model predicting with a 99.76% degree of accuracy.                       |
| Resizing images to a smaller fixed resolution (e.g., 50×50) retains enough visual detail for accurate classification. |          Using smaller image sizes reduces computational cost while preserving key features. The original dataset images are 256x256 pixels, but training with this resolution typically results in a model larger than 100MB—exceeding GitHub’s standard file size limit. Git LFS was avoided by using smaller input dimensions (160x160). This approach maintains performance while keeping the model under the 100MB threshold.                       |
| Average and variability images for each class (healthy vs. infected) will show distinct patterns.|          Model performance shows differences with can be observed in the form of visible white spot/streak patterns. Average images also reveal a color difference where infected leaves appear as a lighter shade of green compared to healthy ones.|
| Using an ML model will reduce manual inspection time by over 90% per tree. | The mildew detector application allows for a near instantaneous prediction of the precense of mildew on a cherry leaf. |

[Back to top](#table-of-contents)

## The rationale to map the business requirements to the Data Visualisations and ML tasks

The business requirements were mapped to the Data visualisations and ML tasks. The accuracy and speed of the results are important along with being scalable and easily understood. The business requirements are:

### Requirement 1

Visual Analysis: conduct a study to visually compare and distinguish between healthy and infected cherry leaves. This requirement can be mapped to the tasks from the following [User Stories](https://github.com/JedDizon/cherryleafmildewdetector-ci-project5/issues?q=is%3Aissue%20state%3Aclosed ):

- User Story - Load dataset and initial analysis
  - As a data analyst, I can load a saved dataset so that I can analyse the data to gain insights on what further tasks may be required.
    - To begin the study, the dataset was downloaded from Kaggle and cleaned by removing any non-image files. The images were then converted to arrays, and average and variability plots were generated for both classes. These visualizations were compared side by side to highlight differences between healthy and infected leaves.
- User Story - Data visualisation
  - As a data scientist, I can visualise the dataset so that I can differentiate a healthy cherry leaf versus one that contains powdery mildew (Business Requirement 1).
  - This page shows the results of the study. It displays to the end user (client) the findings and clearly shows the main difference between healthy and infected leaves. This is seen from the average & variability images, and image montage of healthy and infected leaves.
- User Story - View project analysis on dashboard
  - As a non-technical user, I can view the project findings so that I can receive more detailed information on the analysis done.
  - The user can find the overall analysis of the main difference between healthy and infected leaves.
- User Story - View project hypotheses on dashboard
  - As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.
  - Hypothesis #3 stated that there are distinct visual patterns between healthy and infected leaves.

### Requirement 2

ML-Based Prediction: predicting if a cherry leaf is healthy or infected with powdery mildew based on an image input.This is done via a dashboard that would allow the client to upload images of the leaves and an accurate reading would inform them if the leaves are infected or healthy.

This requirement can be mapped to the tasks from the following [User Stories](https://github.com/JedDizon/cherryleafmildewdetector-ci-project5/issues?q=is%3Aissue%20state%3Aclosed ):

- User Story - Create ML pipeline
  - As a data engineer, I can create an ML model to create a predictor if a cherry leaf is diseased or healthy (Business requirements 2).
  - A convolutional neural network (CNN) was trained to distinguish between healthy and diseased cherry leaves. The model uses multiple layers and activation functions to classify images accurately. It is accessible through the Mildew Detector page on the dashboard.
- User Story - Mildew detector
  - As a non-technical user, I can upload an image of a leaf into the model and receive a prediction if it contains powdery mildew or not (Business requirements 2).
  - On the Mildew Detector page the user can upload images of the leaves. They are run through the model which can predict if the leaves are healthy or are infected with mildew.

[Back to top](#table-of-contents)

## ML business case

Farmy & Foods, an agricultural company, wants to automate the detection of powdery mildew in cherry trees. Currently, this is identified manually by inspecting leaf samples, a process that takes roughly 30 minutes per tree and can be prone to human error.
To address this, a machine learning (ML) model capable of analyzing an image of a cherry leaf and predicting whether it is healthy or infected with powdery mildew was built.

To do this, the following were identified:

- Task Type: Supervised Learning
- Problem Type: Binary classification (Healthy vs. Infected)
- Model Type: Convolutional Neural Network (CNN)
- Input: Image of a single cherry leaf
- Output:
  - A predicted label (Healthy or Infected)
  - Associated probability score
  - Optional visual references (average image per class)

The model is intended to make real-time predictions through a web-based dashboard, where farmers can upload a leaf image and receive instant results.

The success criteria agreed at the start of the project was to have a target accuracy of at least 97%. The prediction time should be near instant and the output should be reliable to help with treatment decisions. Failure to meet the minimum accuracy requirement may result in missed infections, potentially causing significant economic impact.

The dataset used was provided by Farmy & Foods and published on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). It comprises a total of 4208 images that’s made up of healthy and infected leaves. It requires a Kaggle API authentication via JSON key for access. The dataset is split into training, validation, and test subsets. Appropriate precautions are taken to handle the data securely.

If successful, the ML model will:

- Drastically reduce inspection time from 30 minutes per tree to a few seconds

- Improve diagnostic accuracy by reducing human error

- Scale easily across thousands of cherry trees nationwide

- Serve as a proof of concept for similar models across other crops and diseases

[Back to top](#table-of-contents)

## Dashboard Design

Page 1: Project Summary

- General information
  - Powdery mildew is a common fungal disease affecting a wide range of plant species. It is caused by various species of ascomycete fungi and thrives in environments with moderate temperatures and high humidity. In cherry trees, the disease is specifically caused by Podosphaera clandestina, a parasitic fungus that significantly impacts both leaf and fruit quality.
  - The disease manifests as distinctive white, powder-like spots that typically appear on the upper surface of leaves and stems. In the early stages, light-green, circular lesions may form on either side of the leaf. These lesions can develop into a cotton-like fungal growth, especially on newer foliage and, in severe cases, on the fruit itself.
  - This results in:
    - Slowed plant growth
    - Reduced fruit yield
    - Compromised product quality

  - The client, Farmy & Foods, is currently facing a widespread outbreak across several of its cherry tree plantations. The existing method of disease detection involves manual leaf inspection, which is labor-intensive and time-consuming, with each tree taking approximately 30 minutes to assess.
  - To address this challenge, the client has commissioned the development of a Machine Learning (ML) solution capable of automatically detecting signs of powdery mildew from leaf images. The goal is to streamline the inspection process and improve early detection, enabling faster treatment and minimizing crop loss.

- Link to README
  - For additional information, please see the Project [README](https://github.com/JedDizon/cherryleafmildewdetector-ci-project5#readme) file.

- Business requirements
  - The project has two Business Requirements:
    - 1: Enable the client to visually compare and distinguish between healthy and infected cherry leaves using an interactive interface or dashboard.
    - 2: Develop a system that can predict if a cherry leaf is healthy or infected with powdery mildew based on an image input.

- Dataset info
  - The dataset provided contains a total of 4208 images composed of healthy and affected leaves.
  - The dataset is available for download on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).

Page 2: Visual Studies

- Business requirement #1
  - The client aims to visually distinguish between healthy cherry leaves and those affected by powdery mildew (Business Requirement 1).
  - This page showcases the investigation of the visual studies done to answer this requirement.
    - Avg & variability image differences
      - The variability images clearly highlight the distinguishing features between healthy and diseased leaves.
      - The presence of powdery mildew creates visible white spot/streak patterns on infected leaves.
      - Additionally, the average images reveal a color difference where infected leaves appear as a lighter shade of green compared to healthy ones.
  - Average healthy vs infected leaves differences images
  - Image montage

Page 3: Mildew Detector

- Business requirement #2
- This page answers Business Requirement 2.
  - Mildew detector
    - The client asks for an ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew.
  - Link to download set of leaves for prediction
    - A set of healthy and infected leaves for live prediction can be downloaded on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).
  - File uploader widget (Drag & drop)
  - Table with the image name and prediction results & download button

Page 4: Project Hypothesis

- Project hypothesis & validation
  - See [Hypotheses and validation](#hypotheses-and-validation) section.

Page 5: Model performance

- Model Performance Metrics
  - Dataset Splitting Overview
    - The dataset was divided into three subsets: training, validation, and test.
    - This is a common strategy in machine learning for building and evaluating models.
    - Training set: The largest portion of the data, used to teach the model to distinguish between the image classes. A larger training set helps the model learn more effectively.
    - Validation set: Used during model development to fine-tune parameters and improve performance without biasing the model to the test data.
    - Test set: Held back until the very end to provide an unbiased evaluation of the model’s ability to generalize to new, unseen data.
  - Model history
    - The graphs above indicate that the model achieved a consistently high level of accuracy.
    - Performance on the training set improved rapidly during the initial epochs. Although there was a brief dip in accuracy at epoch 4, the model quickly recovered in the following epoch and continued to improve steadily.
    - The loss graph, which measures how far the model’s predictions deviate from the actual values, also reflects strong performance on both the training and validation sets.
  - Performance on Test Set
    - It was requested by the client at the beginning of this project for an ML model with a performance criteria that could predict with at least 97% accuracy if a leaf was healthy or infected with powdery mildew.
    - In the above table, the model predicts with a 99% accuracy on the status of images in the test dataset.
  - Project Conclusions
    - Business Requirement 1 - conduct a study to visually compare and distinguish between healthy and infected cherry leaves.
      - Satisfied. See Page 2 (Visual Studies). Average image of healthy vs infected leaves. Main differences are visible white spot/streak patterns on infected leaves & infected leaves appearing as a lighter shade of green compared to healthy ones.
    - Business Requirement 2 - predicting if a cherry leaf is healthy or infected with powdery mildew based on an image input.
      - Satisfied. See Page 3 (Mildew Detector). Users can upload cherry leaf image(s) and receive a prediction with 99% accuracy if it is healthy or infected with powdery mildew.

[Back to top](#table-of-contents)

## Deployment

- The App live link is: `https://mildew-detector-p5-ci-c8b938269593.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

[Back to top](#table-of-contents)

## Main Data Analysis and Machine Learning Libraries

The main libraries used were:

- numpy 1.26.1 - used for converting to array
- pandas 2.1.1 - used for creating/saving as dataframe
- matplotlib 3.8.0 - used for plotting distribution of the sets
- seaborn 0.13.2 - used for making statistical graphs
- plotly 5.17.0 - used for plotting model learning curve
- Pillow 10.0.1 - used to adjust images
- streamlit 1.40.2 - used for creating the dashboards
- joblib 1.4.2 - used for running tasks in parallel
- scikit-learn 1.3.1 - used for model evaluation
- tensorflow-cpu 2.16.1 - used for model creation
- keras 3.0.0 - used for setting up model hyperparameters

[Back to top](#table-of-contents)

## Other technologies used

- Heroku: Project deployment
- Git/GitHub: Version control
- Gitpod: IDE used for project development
- Jupyter Notebook: develop ML model
- Kaggle: Download dataset
- techsini: Generate a multimockup image of the project

[Back to top](#table-of-contents)

## Issues

Deploying on Heroku: During deployment, the application exceeded the size limit and couldn’t be posted. Although several items were added to the .slugignore file, the size remained too large. To reduce it further, older versions of matplotlib & plotly were used and the Python version was downgraded from 3.12 to 3.9.

`Compiled slug size: 570.1M is too large (max is 500M).`
(Added inputs to slugignore)

`Compiled slug size: 517.7M is too large (max is 500M).`
(Addedjupyter notebooks to slugignore)

`Compiled slug size: 512.3M is too large (max is 500M).`
(python 3.12 → 3.12, matplotlib & plotly versions decreased, added validation images inputs back to allow image montage to run)

`Done: 499.8M`

`!     Warning: Your slug size (499 MB) exceeds our soft limit (300 MB) which may affect boot time.`

[Back to top](#table-of-contents)

## Testing

### Manual Testing

**Requirement 1** - conduct a study to visually compare and distinguish between healthy and infected cherry leaves.

- As a data analyst, I can load a saved dataset so that I can analyse the data to gain insights on what further tasks may be required.

- As a data scientist, I can visualise the dataset so that I can differentiate a healthy cherry leaf versus one that contains powdery mildew (Business Requirement 1).

- As a non-technical user, I can view the project findings so that I can receive more detailed information on the analysis done.

- As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.

| TEST                   | ACTION                                | EXPECTATION                              | RESULT    |
|------------------------|---------------------------------------|------------------------------------------|-----------|
| **Navbar** | Select Visual Studies page button | Visual Studies page opens | **Success** |
| **Average and Variability Image Differences Button** | Click button | Display average & variability image for healthy & infected leaves | **Success**|
| **Average Healthy & Infected Cherry Leaves Differences** | Click button | Display average images & difference image for average healthy & infect leaves | **Success**|
| **Image Montage Button** | Click button | Display dropdown for montage creation | **Success**|
| **Healthy leaves dropdown option** | Select & click button to create montage | See montage of healthy leaves| **Success**|
| **Infected leaves dropdown option** | Select & click button to create montage | See montage of infected leaves| **Success**|

**Requirement 2** - predicting if a cherry leaf is healthy or infected with powdery mildew based on an image input.This is done via a dashboard that would allow the client to upload images of the leaves and an accurate reading would inform them if the leaves are infected or healthy.

- As a data engineer, I can create an ML model to create a predictor if a cherry leaf is diseased or healthy.

- As a non-technical user, I can upload an image of a leaf into the model and receive a prediction if it contains powdery mildew or not (Business requirements 2).

| TEST                   | ACTION                                | EXPECTATION                              | RESULT    |
|------------------------|---------------------------------------|------------------------------------------|-----------|
| **Navbar** | Select Mildew Detector page button |Mildew Detector page opens | **Success**|
| **Kaggle link** | Click on link |Kaggle page for dataset opens | **Success**|
| **Box for uploading images** | Drag & drop leaf image into box | See report displaying analysis of the image | **Success**|
| **Box for uploading images** | Use browse files button | File explorer opens to enable selection | **Success**|
| **Box for uploading images** | Upload image from file explorer | See report displaying analysis of the image | **Success**|
| **Box for uploading images** | Repeat prior two items for multiple images | See report displaying analysis of all the images| **Success**|
| **Image analysis report** | Click button to download csv report of analysis | Report is downloaded containing the results shown on dashboard| **Success**|

## Validation

### PEP8 Validation

[CI Python Linter](https://pep8ci.herokuapp.com) was used to test code for PEP8 compliance.


<hr><summary>app_pages</summary><hr>

<details><summary>multipage.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>

<details><summary>p1_project_summary.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>

<details><summary>p2_visual_studies.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>

<details><summary>p3_mildew_detector.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>

<details><summary>p4_project_hypothesis.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>

<details><summary>p5_model_performance.py</summary>
<img src="docs/pep8/pep8_about_admin.png">
</details>


<hr><summary>src</summary><hr>

<details><summary>data_management.py</summary>
<img src="docs/pep8/pep8_blog_admin.png">
</details>

<details><summary>evaluate_clf.py</summary>
<img src="docs/pep8/pep8_blog_models.png">
</details>

<details><summary>predictive_analysis.py</summary>
<img src="docs/pep8/pep8_blog_urls.png">
</details>

[Back to top](#table-of-contents)

## Credits

- Content
  1. [Streamlit doc](https://docs.streamlit.io/develop/api-reference)
  2. [Streamlit commands](https://github.com/Code-Institute-Solutions/streamlit-lesson/blob/main/1_commands_and_widgets.py )
  3. [Multipageapp](https://github.com/Code-Institute-Solutions/streamlit-calculator)
  4. [CVD Predictor](https://github.com/jfpaliga/CVD-predictor)
  5. [Mildew Detection](https://github.com/Code-Institute-Solutions/milestone-project-mildew-detection-in-cherry-leaves)
  6. [Cherry Picker](https://github.com/HughKeenan/CherryPicker)

- Media
  1. emojis/icons:
     1. [Streamlit Emojis](https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/)

[Back to top](#table-of-contents)

## Acknowledgements

- Code Institute

- Mo Shami (Mentor)

[Back to top](#table-of-contents)
