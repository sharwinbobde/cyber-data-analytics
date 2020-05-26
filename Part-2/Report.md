# CS4035 - Cyber Data Analytics
## Lab 2 

## Group Number : 15

Student 1 
Name : Nikhil Saldanha
ID : 4998707

Student 2
Name : Sharwin Bobde
ID : 5011639

### Instructions for setting up
- Create a folder for the data called `data/`.
- Download the data into `data/` from [here](https://www.batadal.net/data.html).
- Setup virtualenv: `virtualenv -p python3 env`
- Enter virtualenv: `source env/bin/activate`
- Start Jupyter Server: `jupyter notebook .`

## 1. Familiarization task – 1 A4

### 1a. Plot visualizations 

We manually add the attack flag column to the test data from provided information by BATADAL

<img src="/home/sharwinbobde/Pictures/Lab2/2020-03-17_15-18.png" style="zoom:50%;" />

<img src="/home/sharwinbobde/Pictures/Lab2/index.png" style="zoom:50%;" />

<img src="/home/sharwinbobde/Pictures/Lab2/2.png" style="zoom:50%;" />

<img src="/home/sharwinbobde/Pictures/Lab2/3.png" style="zoom: 50%;" />

### 1b. Answers to the three questions

**1. What types of signals are there?**

In the dataset, the signals prefixed with "S" are state variables which are discrete, the signals prefixed with "L" denote the water levels, "F" signals are for flow and "P" signals are for inlet and outlet pressure. We see that each of these signals have their own characteristic shape.

**2. Are the signals correlated? Do they show cyclic behavior?**

From the heatmaps we can clearly see which signals are positively and negatively correlated in both the training datasets. The cyclic nature of the signals can be seen in the individual signal plots and the STFT (Short Term Fourier Transform) plots. while the individual signal plots are in time domain, the STFT is in frequency domain. Thus we can see the characteristic frequencies of the signals through time. A simple way of interpreting them is:

-  prominent horizontal lines: certain frequencies are more prominent therefore is cyclic in nature.
-  prominent vertical lines: signal changes abruptly in a short time.

Thus, we see that *many* signals are cyclic in nature.


**3. Is predicting the next value in a series easy or hard? Use any method from class**

We make an attempt at predicting the next value in the series of the signals L_T1, F_PU2, S_PU1, P_J289 using the sliding window technique. According to our observations, the model can reasonable predict the next value of the signal when there are no sudden and large changes in it. In our visualizations above, we can see that when S_PU1 drops to 0 and then back to 1 in a short interval, the model is not able to predict this. We can see similar patterns in other signals as well. Hence, we would say that whether it is hard or easy depends on the characteristic of the signal itself.

## 2. LOF task – 1/2 A4 – Nikhil Saldanha

### 2a. Plot LOF scores

<img src="/home/sharwinbobde/Pictures/Lab2/4.png" style="zoom: 50%;" /><img src="/home/sharwinbobde/Pictures/Lab2/5.png" style="zoom: 50%;" />

### 2b. Analysis and answers to the questions

**1. Do you see large abnormalities in the training data? Can you explain why these occur?**
Yes, we can see the large abnormalities in the training data through the LOF scores above. The model takes -1.5 as the threshold below which the samples are classified as outliers. These are possibly the datapoints that have sudden changes in their values as we saw in the familiarization task.


**2. It is best to remove such abnormalities from the training data since you only want to model normal behavior?**
We would say that it is not a good idea to remove these from the training data since they are already part of normal behaviour and we should model these kinds of patterns as well.


**3. Describe the kind of anomalies you can detect using LOF**

With LOF you can detect point anomalies since it measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. A point anomaly is defined as: A single instance of data that is anomalous since it deviates largely from the rest of the data points.

## 3. PCA task – 1/2 A4 – Sharwin Bobde (5011639)

### 3a. Plot PCA residuals

<img src="/home/sharwinbobde/Pictures/Lab2/6.png" style="zoom: 33%;" />

### 3b. Analysis and answers to the questions

### Choosing `n_components`
We will use the mean and standard deviation of anomaly score (the probability of a row being an anomaly) to choose. The anomaly score should be low for normal data and high for data we know is outlier. Moreover, we can split `data 2` in two parts (inlier and outlier) to see that it firs the inliers as normal data.

<img src="/home/sharwinbobde/Pictures/Lab2/7.png" style="zoom: 50%;" />

The above plot isjust to show the relation between residual and anomaly score. The mere reduction in residual score does not entail lower anomaly score.

<img src="/home/sharwinbobde/Pictures/Lab2/8.png" style="zoom: 50%;" />

### Select `n_components = 10`
This is because it models the normal data (entire data_1 and data_2 inliers) well with very low residual error AND it shows high anomaly score for data we know is outlier while generalising using less number of components.

### Abnormalities in training data

**Q. Do you see large abnormalities in the trainingdata?**

In the above plot it is quite visible that outlier data has higher probability of being tagged anomalous. We see at `n_components = 31` this observation is most prominant.

**Q. Can you explain why these occur?**

This is because the outlier points (attacks) have abnormal behaviour and their inverse PCA mapping is very erroneous and thus have higher residual. This is because these points have high variance for principal components which usually shouldn't.

**Q. Describe the kind of anomalies you can detect using PCA.**

We can detect outlier points which have a different variance than the regular distribution in the direction of the principal components.

## 4. ARMA task – 1/2 A4 - Nikhil Saldanha

### 4a. Print relevant plots and/or metrics to determine the parameters.

We analyse these 4 signals: "L_T7", "F_PU10", "S_PU10", "S_PU11" for detecting attacks since they were important according to the analysis in the Taormina et.al. paper: **Characterizing Cyber-Physical Attacks on Water Distribution Systems**

We first visualize each of the signals

<img src="/home/sharwinbobde/Pictures/Lab2/9.png" style="zoom: 33%;" />

**To determine the amount of differencing:** The observations in a stationary time series are not dependent on time. Time series are stationary if they do not have trend or seasonal effects. Summary statistics calculated on the time series are consistent over time, like the mean or the variance of the observations. We can see that these signals are all stationary since they do not have an obvious trend. We can even do a test called the Augmented Dickey-Fuller test that can inform us about the degree to which this data is stationary.

The ADF test is a statistical test with the following hypotheses

**Null Hypothesis (H0):** If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
**Alternate Hypothesis (H1):** The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

### NOTE: since ARMA modelling takes quite a while, we continue forward with only 1 signal L_T7 since it the most important (representing water level)

We can see that our statistic values of -17 is less than its corresponding critical values at 1%. This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke).

From the above tests we can conclude that the chosen signal is stationary and we do not need differencing.

**To determine the parameter for the number of AR Terms:** We plot the partial auto-correlation function and see where there is a sharp cutoff with respect to the number of lag terms. According to the figure below, we should use around 2

<img src="/home/sharwinbobde/Pictures/Lab2/10.png" style="zoom: 50%;" /><img src="/home/sharwinbobde/Pictures/Lab2/11.png" style="zoom: 50%;" />

### 4b. Plots to study the detected anomalies

We first do a test run on the train data and see the residuals.

<img src="/home/sharwinbobde/Pictures/Lab2/12.png" style="zoom: 50%;" /><img src="/home/sharwinbobde/Pictures/Lab2/13.png" style="zoom: 50%;" />

We train on the full training data 1 and test on the first 500 samples of the test set 

<img src="/home/sharwinbobde/Pictures/Lab2/14.png" style="zoom: 50%;" />

To classify a point as an anomaly, we maintain a standard deviation of errors in 24 hour windows, between ground truth and predicted. If the deviation is more than 3 standard deviations of usualy error away from the ground truth value, then we classify that as an anomaly

<img src="/home/sharwinbobde/Pictures/Lab2/15.png" style="zoom: 50%;" />

We can see that the model is not that great as it can classify only a few anomalies correctly and while getting too many false positives and also false negatives

### 4c. Analysis and answers to the questions

**1.  What kind of anomalies can you detect using ARMA models?**

The ARMA models can detect contextual anomalies. This is because ARMA models regress on previous values and make next step predictions based on them. A high residual means that the data point in question is anamolous in context to the previous values. The model represents the normal behaviour of the system given those same previous data points and can detect when the current value deviates from this.

**2. Which sensors can be modeled effectively using ARMA?**

Based on experiments, the sensors that correspond to water level and pressure were much easier to model since they showed cyclic behaviour. Flow and status signals are much harder to model since they do not have much of a cycle and jump too quickly from one level to another.

## 5. N-gram task – 1/2 A4 - Sharwin Bobde (5011639)

### 5a. Visualise discretization

<img src="/home/sharwinbobde/Pictures/Lab2/16.png" style="zoom: 50%;" />

Lets select discretization parameters (window_size, levels, overlap) as (2, 5, 1)

### 5b. Analysis and answers to the questions. Also provide relevant plots.

### Selct $L$ and $N$ now

<img src="/home/sharwinbobde/Pictures/Lab2/17.png" style="zoom:50%;" />

### Explaination
After discretization I tried combinations of L and N. For every block L I calculated N-Gram frequencies and made a profile signature. I found matching labels for this too. I used dataset 2 to characterise normal and anomalous signatures. Using dataset 1 causes heavy oversampling of the normal class.

Clearly there are a lot of things to optimise:
window_size, discretization levels, overlap, L, N, k in k-NN, minory class sampling... thus it proved to be quite difficult to optimise. Looking at the AUC the k-NN is clearly biased and always classifies as either always all 1s or all 0s

## 6. Comparision task 1 A4

### 6a. Use the given guidelines and provide a comparison of the above implemented methods.

### The 4 Methods
We will train and test the 4 methods one by one, save the results in `results{}` and show all the results together.

<img src="/home/sharwinbobde/Pictures/Lab2/18.png" style="zoom:50%;" />

<img src="/home/sharwinbobde/Pictures/Lab2/19.png"  />

<img src="/home/sharwinbobde/Pictures/Lab2/20.png" style="zoom: 80%;" />



### Explanation
We think that the confusion matrix is the best way to evaluate the different models since it allows us to see positives broken down into true positives and false positives. None of the methods we implement show great results but, LOF seems the best achieving normalised TP of 0.05 while keeping the FP to 0.07. The confusion matrix also allows us to see the False Negatives at the same time. FNs are important to keep low since it means that there is a breach in the system that we could not recognise which is more dangerous than having FPs.

According to our results, LOF seems the best strategy but we could definitely achieve better results if we could combine these methods to detect many different kinds of anomalies. PCA strongly came in second, because of its ability to generalise and reconstruct the signal well.

N-Gram and ARMA definitely performed the worst. ARMA was unable to model the signal well. It may require some more tuning to give good performance. Regarding N-gram, even though it is able to generate signatures of the data properly the classification mechanism with nearest neighbour method is giving high bias and it always outputs either of 0 or 1. It will require extensive hyperparameter optimisation to work properly or better study of the original signal.