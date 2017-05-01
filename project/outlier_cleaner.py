#!/usr/bin/python


def outlierCleaner(predictions, features, labels):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error=(labels-predictions)**2
    cleaned_data = zip(error, features, labels)
    cleaned_data= sorted(cleaned_data, key=lambda x: x[0], reverse=True)
    limit = int(len(labels)*0.1)
    #print("cleaned data is", cleaned_data)
    return cleaned_data[limit:]

