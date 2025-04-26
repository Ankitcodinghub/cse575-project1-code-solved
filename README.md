# cse575-project1-code-solved
**TO GET THIS SOLUTION VISIT:** [CSE575 Project1 Code Solved](https://www.ankitcodinghub.com/product/cse575-project1-code-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;69846&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE575 Project1 Code Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
In this class project, we will systematically implement and examine the three major categories of Machine Learning techniques of this course, including supervised learning, un-supervised learning, and deep learning.

<h1>Part 1: Density Estimation and Classification</h1>
<strong>Project Overview: </strong>

<strong>Preparation</strong>

First of all, when you open your lab, you will be in the ‘renameNB’ Jupyter notebook. As you run the code, you will load the trainset and testset for digit0 and digit1 respectively (Please read code and you will understand). Both trainset and testset are sub-dataset from the MNIST dataset. The MNIST dataset contains 70,000 images of handwritten digits, divided into 60,000 training images and 10,000 testing images. We use only a part of images for digit “0” and digit “1” in this question.

Therefore, we have the following statistics for the given dataset:

Number of samples in the training set:&nbsp; “0”: 5000 ;”1″: 5000.

Number of samples in the testing set: “0”: 980;&nbsp;&nbsp; “1”: 1135

We assume that the prior probabilities are the same (P(Y=0) = P(Y=1) =0.5), although you may have noticed that these two digits have different numbers of samples in testing sets.

In the existing code, myID is a 4 digit string and please change this string to your last 4 digit of your own studentID; train0 is your trainset for digit0; train1 is your trainset for digit1; test0 is your testset for digit0; and test1 is your testset for digit1. They are all Numpy Arrays. You can also convert them into python arrays if you like.

Other than the string named ‘myID’, <strong>please DON’T</strong> change any existing code and just write your own logic with the existing code.

You may go to the original MNIST dataset (available here<a href="http://yann.lecun.com/exdb/mnist/"> http://yann.lecun.com/exdb/mnist/</a><a href="http://yann.lecun.com/exdb/mnist/">)</a> to extract the images for digit 0 and digit 1, to form the dataset for this project. To ease your effort, we have also extracted the necessary images, and store them in “.mat” files. You may use the following piece of code to read the dataset:

import scipy.io

Numpyfile= scipy.io.loadmat(‘matlabfile.mat’)

<strong>.mat files for you to download: </strong>Attachment <strong>mat files pt 1</strong>

ZIP File

Download Trash

<strong>Programming</strong>

For your own code logic, you have 4 tasks to do:

<strong>Task 1</strong>. You need to first extract features from the your original trainset in order to convert the original data arrays to 2Dimentional data points.

You are required to extract the following two features for each image:

Feature1:The average brightness of each image (average all pixel brightness values within a whole image array)

Feature2:The standard deviation of the brightness of each image (standard deviation of all pixel brightness values within a whole image array)

We assume that these two features are independent, and that each image is drawn from a normal distribution.

<strong>Task 2</strong>. You need to calculate all the parameters for the two-class naive bayes classifiers respectively, based upon the 2D data points you generated in Task1. (Totally you should have 8 parameters)

(No.1) Mean of feature1 for digit0

(No.2) Variance of feature1 for digit0

(No.3) Mean of feature2 for digit0

(No.4) Variance of feature2 for digit0

(No.5) Mean of feature1 for digit1

(No.6) Variance of feature1 for digit1

(No.7) Mean of feature2 for digit1

(No.8) Variance of feature2 for digit1

<strong>Task 3</strong>. Since you get the NB classifiers’ parameters from Task2, you need to implement their calculation formula according to their Mathematical Expressions. Then you use your implemented classifiers to classify/predict all the unknown labels of newly coming data points (your test data points converted from your original testset for both digit0 and digit1). Thus, in this task, you need to work with the testset for digit0 and digit1 (2 Numpy Arrays: test0 and test1 mentioned above) and you need to predict all the labels of them.

PS: Remember to first convert your original 2 test data arrays (test0 and test1) into 2-D data points as exactly the same way you did in task1.

<strong>Task 4</strong>. In task3 you successfully predicted the labels for all the test data, now you need to calculate the accuracy of your predictions for testset for both digit0 and digit1 respectively.

The specific algorithmic tasks you need to perform for this part of the project include:

<ol>
<li>Extracting the features and then estimating the parameters for the 2-D normal distribution for each digit, using the training data. Note: You will have two distributions, one for each digit.</li>
<li>Use the estimated distributions for doing Naïve Bayes classification on the testing data. Report the classification accuracy for both “0” and “1” in the testing set.</li>
</ol>
<strong>Algorithms:</strong>

MLE Density Estimation,&nbsp; Naïve Bayes classification

<strong>Resources:&nbsp; </strong>

A subset of MNIST dataset, download either from<a href="http://yann.lecun.com/exdb/mnist/"> http://yann.lecun.com/exdb/mnist/</a> (requiring you to extract data corresponding to digit 0 and digit 1 only),&nbsp; or from the .mat files provided.

<strong>Workspace: </strong>

Any Python programming environment.

<strong>Software: </strong>

Python environment.

<strong>Language(s): </strong>

Python.

<strong>Result Submission</strong>

As the result from your Jupyter Notebook of project1, you should have 8 components for computed parameters and 2 components for accuracy. The order of these 10 components should look like this:

[Mean_of_feature1_for_digit0, Variance_of_feature1_for_digit0,

Mean_of_feature2_for_digit0, Variance_of_feature2_for_digit0

Mean_of_feature1_for_digit1, Variance_of_feature1_for_digit1

Mean_of_feature2_for_digit1, Variance_of_feature2_for_digit1,

Accuracy_for_digit0testset, Accuracy_for_digit1testset]

When you get all the components for your answer, please go to ‘Quiz: Project Part1(Window for submission)’ and do the submission there.

PS: You <strong>CANNOT </strong>submit anything in your Jupyter Notebook. In this project, Jupyter Notebook is simply a uniformed IDE we provide you to implement the project.

&nbsp;

<strong>Grading Criteria</strong>

1 point for mean and variance of f1 for digit0

1 point for mean and variance of f2 for digit0

1 point for mean and variance of f1 for digit1

<ul>
<li>point for mean and variance of f2 for digit1</li>
<li>points for predicting new labels for digit0testset and calculating the accuracy.</li>
</ul>
2 points for predicting new labels for digit1testset and calculating the accuracy.

PS: The <strong>acceptable</strong> range for parameters is [x-0.2, x+0.2]; The <strong>acceptable</strong> range for accuracy is [x-0.005, x+0.005]. It means that if one of your float-number answer falls into its corresponding range, your answer will be graded as correct. No, otherwise.
