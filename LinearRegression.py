from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Makes sure your chart shows in the jupyter newbook

medical_df = pd.read_csv(r'c:\Users\pm0443\Downloads\insurance.csv',low_memory=False)
medical_df.head(5)

#value counts
val_count = ['sex','smoker','region','children']
for col in val_count:
    print ("Value count for column : " , col)
    print (medical_df[col].value_counts())
    print('\n')


print("Correlation with charges ")
print("Charges vs Age "  , medical_df.charges.corr(medical_df.age))
print("Charges vs BMI"  , medical_df.charges.corr(medical_df.bmi))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

#Convert all categorical variables to numerical
#{ 0 : female , 1 : male}
le = LabelEncoder()
label = le.fit_transform(medical_df['sex'])
medical_df.drop("sex", axis=1, inplace=True)
medical_df["sex"] = label

#{ 0 : no , 1 : yes}
le = LabelEncoder()
label = le.fit_transform(medical_df['smoker'])
medical_df.drop("smoker", axis=1, inplace=True)
medical_df["smoker"] = label

#{southeast : 2 , southwest : 3 , northwest : 1 , northwest : 0 }
le = LabelEncoder()
label = le.fit_transform(medical_df['region'])
medical_df.drop("region", axis=1, inplace=True)
medical_df["region"] = label

#correlations
print("Correlation with charges ")
print("Charges vs Region "  , medical_df.charges.corr(medical_df.region))
print("Charges vs Sex"  , medical_df.charges.corr(medical_df.sex))
print("Charges vs Smoker"  , medical_df.charges.corr(medical_df.smoker))

medical_df.corr()

sns.heatmap(medical_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');

#non smokers

non_smoker_df = medical_df[medical_df.smoker == 0]
plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15);

def estimate_charges(age, w, b):
    return w * age + b # linear regression equation
#suppose
w = 50
b = 100
ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)

plt.plot(ages, estimated_charges, 'r-o');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');


target = non_smoker_df.charges 

plt.plot(ages, estimated_charges, 'r', alpha=0.9);
plt.scatter(ages, target, s=8,alpha=0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);


# red line is base on the parameters that we gave for w and b
# We have to increase the weight
#Gets weight and target and thenn we can call t mutple w and b values

    #Point is to reduce the gap between the best fit line and the scatter points

def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)
    
    plt.plot(ages, estimated_charges, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual']);


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))
  
targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w, b)


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)
    
    plt.plot(ages, predictions, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)
 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
help(model.fit)

inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)

model.fit(inputs, targets)

model.predict(np.array([[23], 
                        [37], 
                        [61]]))

predictions = model.predict(inputs)


rmse(targets, predictions)









