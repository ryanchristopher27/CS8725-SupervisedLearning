from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
# Grab only Setosa and Versicolour
X = iris.data.features[:100]
y = iris.data.targets [:100]


  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 
