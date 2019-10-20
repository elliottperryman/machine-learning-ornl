# machine-learning-Ca45
This is all the machine learning code I've used for Nab and Ca45 data.
If you want to just jump in, check out 
* DNP_2019 for the most recent code
* data-exploration for all my data-exploration code
* others for more side problems (optimized synthetic data generation, optimal proprocessing filtering, etc)

# Organization
Since this is a comprehensive repository of all the machine learning 
I have done with Ca45/Nab data (the data is very similar), there are 
several different purposes all togther:
 * data exploration - using real data, try to find what is in it. the goals here are
    * what is in the data?
    * what are fast/accurate ways to select different features?
 * pileup identification - this answers several questions:
    * what is the optimal preprocessing for training ML to recognize two events in the same readout (pileup)?
    * what architecture performs best?
    * what is the best structure for the output?
    * how can i compare different methods?
    
    
Many of these use a few similar things:
 * synthetic data generators
 * traditional signal processing methods
 * data wrangling - converting from a custom binary data type, chopping off uninformative data, etc
 * preprocessing -- normalizing, making cuts to the data, etc
 * keras/sklearn toolkits
 
 
    
