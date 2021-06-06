# K-means
In this assignment I have written my own K-means code and investigate the performance of different kernel methods, where the objective was to minimize the following equation: 

![image](https://user-images.githubusercontent.com/48417171/120928487-7d7fac00-c6ed-11eb-94fd-bf7c5967ec50.png)

For this assignment xs’ correspond to images and cs’ are the centroids. Furthermore, I have used 3 additional different kernels as stated in the homework: 

![image](https://user-images.githubusercontent.com/48417171/120928492-84a6ba00-c6ed-11eb-9d53-d0f8dc04f303.png)

Using these Kernel has changed the equation we want to minimize as following:

![image](https://user-images.githubusercontent.com/48417171/120928498-8c665e80-c6ed-11eb-990c-939bfde5cf02.png)

Where K is kernel matrix and W matrix holds the information of which image belongs to which class. Lastly, to test this implementation, I have used the MNIST dataset where the train set has 60000 and the test set has 10000 image samples.

It is important to note that because creating kernel matrixes was computationally heavy and I, unfortunately, have low computational power. For tasks of this assignment, I have randomly sampled the train set. I was carefully sampling the matrix where at the sampled matrix there was equal number of images correspond to different classes. 

## Hyperparameter Tuning

For hyperparameter tuning, I have used 5000 sized train set (500 images for each class). Also, I have used 5000 sized validation set where I took from the unused side of the train set. To tune the hyperparameters I have first run the K-means until convergence (when the assigned classes did not change between iterations I decided that classifier has converged) for the train set and then used the learned centroids for inference on the validation set. Mathematically after doing training, inference on the validation set can be done by calculating 2 more Kernels:

![image](https://user-images.githubusercontent.com/48417171/120928516-9f792e80-c6ed-11eb-89c9-db9c5cdc9152.png)

Where A ̂ and B ̂ correspond to training and validation set matrixes mapped by the kernel function, respectively. For comparison, I have used accuracy as the metric. Since K-means is unsupervised learning, the training result has random labels. For instance, although the K-means is able to distinguish images of “0” from other images, because we did not force the algorithm to label it as “0” it can label it to any value between 0 and “k” (number of classes). Therefore, to find the accuracy I have corrected the learned labels first (statistically checked which k-means label correspond to which ground truth label) and then find the accuracy. 

For tuning, I have used the grid-search technique and because there are 2 parameters to tune it is not possible to show the results on 2D-plot (1- accuracy, 2-hyperparameter[0], 3-hyperparameter[1]) therefore I have used print() function and looked at the best combination that gives the highest validation set accuracy. For justification, I will copy-paste the screenshots of print() results.

To find the best hyperparameters of the “Polynomial” kernel I have first done some experiments and decided to run a grid search for the values of “c_poly” (constant in the kernel) with “d” (power) as linspace(-25, 25, 51) and linspace(1, 5, 5) respectively. I have seen that increasing power decreases accuracy enormously. As a result of this search, I have found the best parameters as following.

![image](https://user-images.githubusercontent.com/48417171/120928528-ad2eb400-c6ed-11eb-878d-9cae5cdcc8be.png)

![image](https://user-images.githubusercontent.com/48417171/120928531-aef87780-c6ed-11eb-8134-0134949c9474.png)

I cannot say this is the most optimum parameter for this set because I have found these parameters by sampling the original dataset. However, we can say that this value is close enough to the most optimum parameters. 

I have repeated this procedure both for the “Gaussian” and “Sigmoid” kernels. For the “Sigmoid” kernel after some first examinations, I have done a grid search for “c_sig” (constant in the kernel) with “theta” as linspace(0.1, 1, 10) and linspace(-10, 10, 21) respectively. As a result of this search, I have found the best parameters as c_sig=0.1,theta=-5.

![image](https://user-images.githubusercontent.com/48417171/120928540-b9b30c80-c6ed-11eb-9877-f02317f1c3e4.png)

For the “Gaussian” kernel after some first examinations, I have done a grid search for “std” (standard deviation in the kernel) as linspace(0.4, 10, 97). As a result of this search, I have found the best parameters as following.

![image](https://user-images.githubusercontent.com/48417171/120928547-bddf2a00-c6ed-11eb-9699-1af06fb82a31.png)

As previously said, I cannot say that this is the most optimum parameter for this dataset since these parameters were found by sampling the initial dataset. We may, therefore, assume that this value is reasonably close to the optimum parameters or these parameters are local minima’s.


## Identifying the Number of Classes 

By using the hyperparameters found in the previous section (we do not have to use those parameters in this section actually but I did), I have calculated the equation we are minimizing for different k-values, the equation we are minimizing is as following:

![image](https://user-images.githubusercontent.com/48417171/120928565-d3ecea80-c6ed-11eb-9ed7-83eb999a45f6.png)

Let us call this loss function. I have found this loss function for different k values for different kernels and plotted the result. Since there are 10 different classes what we should see is that an “elbow” in the graph at k=10. In other words, after k gets bigger than 10 we should see that the loss does not decrease as it decreases for the values smaller than 10. The results are as follows.  

![image](https://user-images.githubusercontent.com/48417171/120928570-d8b19e80-c6ed-11eb-977c-eea82bd65b92.png)

![image](https://user-images.githubusercontent.com/48417171/120928573-db13f880-c6ed-11eb-9ca6-45a8701370d5.png)

![image](https://user-images.githubusercontent.com/48417171/120928576-dcddbc00-c6ed-11eb-925d-861c69d9b944.png)

![image](https://user-images.githubusercontent.com/48417171/120928582-dea77f80-c6ed-11eb-87ea-8b5cbb726290.png)
  
Note that I have used 10000 sampled matrixes for these graphs and as we can see that there are no clear elbows in these graphs. By seeing these I have understood that k-means for any type of kernels is not able to distinguish classes clearly for this dataset. I will further discuss and show proof that the k-means is not able to distinguish different classes in the next section. I have proceeded to the next section by choosing k=10. Also, while doing these calculations it was taking 10-60 iterations for k-means to converge. I will also share the final converge iteration number for k=10 in the following section. 


## Accuracy for Train and Test Sets

For this part by using the hyperparameters found in the previous section I have trained and found the centroids, iteration number until converge, and the accuracy on the sets as following. For this result, I did not divide the train set into two parts and used the 1/6 training set (10000 images) for training and the whole test set for the test. The results are as follows.

![image](https://user-images.githubusercontent.com/48417171/120928596-eebf5f00-c6ed-11eb-8ad7-986dcc11db35.png)

![image](https://user-images.githubusercontent.com/48417171/120928599-f121b900-c6ed-11eb-8bc6-fa285af341be.png)

![image](https://user-images.githubusercontent.com/48417171/120928603-f3841300-c6ed-11eb-8af6-8b6ad322ac60.png)

![image](https://user-images.githubusercontent.com/48417171/120928607-f4b54000-c6ed-11eb-8747-d9035b24e519.png)

![image](https://user-images.githubusercontent.com/48417171/120928611-f67f0380-c6ed-11eb-9112-f37cf6e16ae7.png)

First of all, it is important to look at the centroids of k-means classifiers to interpret why we are getting accuracy around %50. We can see that hyperparameter tuning is done well because the centroids look clear. Also, from the centroids, we can easily see that the k-means classifier is not able to distinguish all the classes because some centroids correspond to the same numbers/classes. For example, if we look at the Gaussian kernel centroids (Figure 9), we can see that there are 3 different centroids that look like the number 9. After consideration, I think that one of these centroids is also responsible for detecting 4. The centroid in the second-row second-column looks like a bit to 4. Therefore, I think that centroid is a mixture of 4’s and 9’s. Another better example is first-row first-column. It looks like a mixture of “5” and “0”. Another missing number is “7”, but we may think that “7” and “9” look similar. Thus, I reckon that k-mean is not able to distinguish 9s’, 7s’ and 4s’, and in the result centroids there are 3 centroids that are a mixture of these numbers that ends up looking like “9”s. 

Because the k-means is not able to distinguish numbers clearly the accuracy is below %60 generally. Furthermore, there is not much difference between the training set and test set results. This shows that our model is not overfitting in the training set. Mostly, the test accuracy is higher than training set accuracy, I think we are getting this result just because the test set has images of the numbers that are more distinguishable than the training set, i.e., by luck, for the k-means test set is easier to classify. 

When I compare different kernels I have seen that the polynomial Kernel is the best one whereas the sigmoid kernel works the worst. I am not surprised by this result because in the coursebook it is stated that for the visual tasks Polynomial kernel is highly used because of its success whereas sigmoid is generally used with neural networks, not with k-means. The result of the gaussian and “Non-kernel” (normal Euclidian distance) looks similar. In the book, it is stated that Gaussian kernel is highly used in statistics, for this vision task I have experienced no accuracy performance increase from Gaussian kernel. 

The result of the number of iterations required for the classifier to converge for different kernels is also proportional to their accuracy result. The polynomial kernel is converging fastest with only 31 iterations (10 k images in the training set) whereas the “non-kernel” method and Gaussian kernel converge at around 40 iterations and sigmoid converges after 80 iterations. I concluded that the better the kernel can classify the faster it converges. 

Finally, because for initialization I assign each image instance to a random class, the result of the training and test set accuracies comes different for every try. However, in general, I saw similar results. 
