# Prague housing analysis
This repo contains a mini-project dedicated to analyzing the data from a Czech website (www.sreality.cz) dedicated to aggregating for-sale propositions of arpartments and property. The goal is to train a model capable of producing an "educated guess" on the fair price of a given apartment. The next logical step would be to compare the estimated price with the actual price asked by the owners or their real estate agents to identify potentially overpriced or underpriced bids.<br>

General approach taken:
1. Take the scraped apartment information and price (code for this can be found here https://github.com/dbrovml/my-scrapers/tree/main/sreality)
2. Extract clean features / apartment attributes, e.g. area, location, planning, etc.
3. Train a regression model mapping apartment attributes to its asked price
4. Score the model on the apartments in the dataset
5. Compare the price predicted by the model with the actual price asked by the seller
6. Present a shortlist of properties that seem to be underpriced

Key facts:
1. Hypothesis: features and attributes of an apartment should determine its fair price, i.e. apartments are priced in a relatively objective manner with respect to their material  value.
2. Data: the dataset consists of 5 547 apartments for sale in Prague. The description of 25 apartment attributes (features) can be found in the feature extraction notebook.
3. Model: LightGBM regressor. This choice was made due to the relatively small dataset at hand and due to the fact tree-based model seem to be appropriate from the design point of view. If the hypothesis is correct, apartments with similar attributes should have similar price. Tree models split the feature space in a supervised manner and place similar observations in same node leaves. If an apartment's price deviates from the mean price in its leave too much, it can be marked as potentially mispriced.
4. Training protocol: small dataset allows (and requiries) K-fold validation tuning. Hyperopt was used to perform Bayesian hyperparameter search for hyperparameter tuning.

Results and potential improvements:
1. Out of 5 547 apartments in the dataset, 648 were identified as potentially underpriced.
2. The model could be improved by pooling more observations from other websites.
3. In case a larger dataset could be comprised, unstructured inputs, such as apartment description and pictures could be leveraged as well.

The code:
1. `features.ipynb` - cleans and pre-processes the scraped apartment features.
2. `exploration.ipynb` - performs a quick-and-dirty EDA to identify potential critical problems.
3. `model.ipynb` - tunes a LightGBM regressor and compares its predictions to actual asked prices.