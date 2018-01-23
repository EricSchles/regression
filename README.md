# Regression - a general solving routinue for regression problems

The goal of this general procedure, will be simple, to analyze data and then generate a statistical model that best fits the data.  The procedure will be so general, that a good enough model is guaranteed, however it may not be the best possible model.

How does the regression module work?

1. Analysis of error

2. Ensembling of Models

3. Classification of new values

The goal of this module is to understand signals in our data and tune our model accordingly.  In the digital signal processing realm there is something called the fundamental frequency.  I believe this idea can be translated into the general regression context as the fundamental model, with other models acting to fill in the gaps.

Broadly speaking data should be classifiable into two categories, those well fit by the model and those not well fit by the model.  Where "the model" is loosely defined and may in actuality be a set of models rather than one well defined model.

Examples of strict models will be things like:

* Linear Regression
* K-nearest Neighbors
* Support Vector Machine
* Decision Tree
* Simulated Annealing

Examples of loosely defined models will be things like:

* Neural Networks
* K-means with Linear Regression applied to each class in K classes.
* Genentic Algorithms
* Random Forrests
* Gradient Boosted Trees
* General Ensembles
* Stacked Ensembles

Once the model, whatever form it may take, is fit to the data the model's errors will be analyzed.  The errors will be calcuated via trimean root squared error and trimean absolute error.  Data that is considered an outlier under either or both descriptive statistics will take a number of possible courses.  Outliers will be detected via the following procedure:

```
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the trimean absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    center = trimean(points)
    diff = np.sum((points - center)**2, axis=-1)
    diff = np.sqrt(diff)
    center_abs_deviation = center(diff)

    modified_z_score = 0.6745 * diff / center_abs_deviation

    return (modified_z_score > thresh).any()
```

Regarding the possible courses of action for outliers:

1. If the number of outliers is seen to be small relative to the overall data and the amount of new data that is generated is also relatively small, then the outliers are simply thrown out.

To make this example concrete - Say you have 100,000 rows of data, 5 outliers (that's 0.005 percent of the data) and a maximum of 1,000 data points generated per month.  The chances of these 5 outliers being meaningful is:

Probability of encountering an outlier := 5/100,000 -> 0.00005

Maximum number of new data ponts := 1,000

Likely number of new outliers in new data = 0.00005 * 1000 = 0.05

So it is very unlikely that an outlier will be found in new data, once next month's data comes in.  Say expanded our search to the next twelve months of data:

Likely number of new outliers over next 12 months = 0.00005 * 12000 = 0.6

That means over the course of the next year, you won't find a single outlier.  For most cases, this means the data is safe to throw out.  Part of the reason for this is simple, most models only survive for a few months anyway.  So these chances seem reasonable, given the specifics.

2. Of course, now we can assume another case, to illustrate when we would want to hold onto outliers:

Assume we are in the same starting conditions of 100,000 rows of data and 5 outliers.  Now let's assume a maximum of 100,000 data points generated per month.  The chances of finding an outlier remain the same:

Probability of encountering an outlier := 5/100,000 -> 0.00005

However,

Maximum number of new data points := 100,000

Likely number of new outliers in new data = 0.00005 * 100,000 = 5.0

In other words, every month there are likely to be 5 new outliers, which is not as good.  Depending on the context you are working in, this might be a big deal.  So we can deal with these outliers explicitly by storing them in a seperate model and explicitly modeling for them.

First, we classify the data as likely outlier or likely not outlier based on the independent outlier data we do have.  Keep in mind, we have to have generated enough outlier data, in order for this classification step to be reasonable.  Then regress the majority of the data over the fundamental model and regress the outliers over the outlier model.

In this way, we can keep track of both typical and exceptional events and determine how often each will occur and what circumstances to watch out for, for exceptional events.


To summarize our considerations:

1. How often new data is generated
2. How long we expect to use our model
3. How much data we have to begin with for training and testing

Using these three meta parameters we are able to distinguish meaningful outliers versus non-meaningful outliers.
