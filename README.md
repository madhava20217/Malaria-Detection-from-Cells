# Malaria Detection using Cell Images

## How to Run the Code

- Prepare the data by using the data_download.ipynb notebook found in the 'Data Download' directory.
  - Tune the required height and width (parameters at the top of the notebook)
  - The output should create a Data directory containing the original cell images, and a Resized_data_<width><height> directory, containing the resized images.
- Label the data using the labelling.ipynb notebook found in the 'Data Labelling' directory.
  - It will save a CSV of relative filenames and labels in the specified directory.
- Create train and test splits using train_test_split.ipynb
- Modeling scripts are in the 'Modeling' directory.


## Contributors
<ol>
    <li>Srishti Singh, srishti20409@iiitd.ac.in</li>
    <li>Shreya Bhatia, shreya20542@iiitd.ac.in</li>
    <li>Madhava Krishna, madhava20217@iiitd.ac.in</li>
    <li>Harshit Goyal, harshit20203@iiitd.ac.in</li>
</ol>

## Motivation

Malaria is a life-threatening disease affecting many people wordwide, spread by infected *Anopheles* mosquito bites. Earlier studies have shown that the degree of agreement between physicians on the acuteness of the disease in a given patient's sample is very low. Preliminary detection aided by computer systems can be of utmost importance for faster and reliable diagnosis. We aim to create a classifier for paratisized and non-parasitized cells to aid medical professionals in this venture.

## Related Work

<ul>
    <li>Pan, et al. (2018) created a model based on deep CNN architectures. They were able to obtain accuracies of over 90% on the training and validation samples using data augmentation.</li>
    <li>Raihan and Nahid (2021) created a model based on boosted trees with feature engineering and determined feature importance using Shapely Additive Explanations (SHAP).</li>
    <li>Fuhad et al. (2020) implemented a CNN based model with accuracy over 99% while being computationally efficient.</li>
</ul>

## Suggested Outcomes

Automation of the diagnosis process will guarntee accurate diagnosis and, as a result, holds the possibility of providing dependable healthcare to places with limited resources. We aim to implement various algorithms for classification while attempting to find optimal parameters for optimising training time, computational complexity and performance. We will attempt transformations and feature engineering and extraction on the dataset. We are going to apply various machine learning models such as SVMs, logistic regression, decision trees, random forest, and compare the performance of all models. We intend to also attempt grayscale conversion and observe the change in behavior of the models.

<hr>

# Project Proposal

<object data="Project_proposal_group_17.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="Project_proposal_group_17.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Project_proposal_group_17.pdf">Download PDF</a>.</p>
    </embed>
</object>
