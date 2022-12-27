# Prague housing prices analysis
This repo contains a mini-project dedicated to analyzing the data from a Czech [resource](www.sreality.cz) dedicated to aggregating for-sale bids of arpartments and real estate. The goal is to train a model capable of producing an "educated guess" on the fair price of a given apartment. The next logical step would be to compare the estimated price with the actual price asked by the owners or their real estate agents to identify potentially overpriced or underpriced bids.

<b>General approach taken</b>:
1. Take the scraped apartment information and their prices (code for this can be found [here](https://github.com/dbrovml/my-scrapers/tree/main/sreality))
2. Extract clean apartment features, e.g. area, location, planning, amenities, etc.
3. Train a regression model mapping apartment attributes to its asked price.
4. Compare the predicted price with the actual price asked by the seller.
5. Make a shortlist of apartments that seem to be underpriced.

<b>Key facts</b>:
1. *Hypothesis*: features and attributes of an apartment should determine its fair price, i.e. apartments are priced in a relatively objective manner with respect to their material  value.
2. *Data*: the dataset consists of <b>5 547</b> apartments for sale in Prague. The description of <b>25</b> apartment features can be found in the feature extraction notebook.
3. *Model*: LightGBM regressor. This choice was made due to the relatively small dataset at hand and due to the fact a tree-based model seems to be appropriate from the design point of view. If the hypothesis is correct, apartments with similar attributes should have similar prices. Tree models split the feature space in a supervised manner and place similar observations in the same node leaves. If an apartment's price deviates from the mean price in its leave too much, it can be marked as potentially mispriced.
4. *Training protocol*: our relatively small dataset allows and requires a K-fold validation protocol. Hyperopt was used to perform Bayesian hyperparameter search for hyperparameter tuning.

<b>Results and potential improvements</b>:
1. Out of <b>5 547</b> apartments in the dataset, <b>648</b> were identified as potentially underpriced.
2. The model could be improved by pooling more observations from other websites.
3. Unstructured inputs, such as apartment description, could be leveraged as well.

<b>Code</b>:
1. `features.ipynb` - cleans and preprocesses the scraped apartment features.
2. `exploration.ipynb` - performs a quick-and-dirty EDA to identify potential critical problems.
3. `model.ipynb` - tunes a LightGBM regressor and compares its predictions to the actual asked prices.