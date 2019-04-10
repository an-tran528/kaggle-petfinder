# PetFinder.my Adoption Prediction 

"In this competition you will predict the speed at which a pet is adopted, based on the pet’s listing on PetFinder. Sometimes a profile represents a group of pets. In this case, the speed of adoption is determined by the speed at which all of the pets are adopted. The data included text, tabular, and image data. See below for details.
This is a Kernels-only competition. At the end of the competition, test data will be replaced in their entirety with new data of approximately the same size, and your kernels will be rerun on the new data. "
    
(more info: https://www.kaggle.com/c/petfinder-adoption-prediction/data) 

## Overall result
Ranked 13th/2010 in Public board with Quadratic weighted kappa score (QWK) 0.481 (Private board: 30/2010)  
The kernel runs 19400s on Kaggle's CPU  
        model name	: Intel(R) Xeon(R) CPU @ 2.30GHz  
        cpu MHz		: 2300.000  
        cpu cores	: 16  

# Methods
-Metadata features: We simply extract basic statistics features like sum, mean, min, max of the metadata   
  
-NLP: both models use fastText and train on the given text data with some preprocessings. We did not used any pretrained word vectors
    
-CV: pretrained DenseNet121 model (trained on ImageNet)
    
In 1 of our models, we feed at most 5 images of a pet through DenseNet and used PCA to reduced dimensionality  
-> This boosted our scores by 0.01 QWK 
    
-Some features engineering: Levenshtein distance of original and processed texts (done by Google Natural Language API); Image's size, vertices, dimension...
  
Apply target mean-encoding for some categorical data like Breed, Color and state
    
-Models: Simply blending 2 Light-GBM models with coefs 0.5-0.5. The models are pretty similar but 1 focuses on text data and 1 focuses on image data

### Have fun Kaggling!
