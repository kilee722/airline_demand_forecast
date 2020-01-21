#Project

import pandas as pd
import numpy as np


def airlineForecast(trainingDataFileName,validationDataFileName):

    #reading files and convert to DF
    training = pd.read_csv(trainingDataFileName)
    validation = pd.read_csv(validationDataFileName)
    trainingDF = pd.DataFrame(training)
    validationDF = pd.DataFrame(validation)
    
    #creating days prior column and day of week column
    trainingDF['booking_date'] = pd.to_datetime(trainingDF['booking_date'])
    trainingDF['departure_date'] = pd.to_datetime(trainingDF['departure_date'])
    trainingDF['Days Prior'] = (trainingDF['departure_date'] - trainingDF['booking_date']).dt.days
    trainingDF['Day of Week'] = trainingDF['departure_date'].dt.weekday_name
    
    #create future demand column (total)
    df1 = pd.DataFrame(trainingDF.departure_date.unique(), columns =['departure_date'])
    df2 = pd.DataFrame(trainingDF['cum_bookings'].loc[trainingDF['Days Prior'] == 0].values, columns = ['Future Demand(Total)']) #did it this way so I didnt have to rename
    trainingDF = trainingDF.merge(pd.concat([df1,df2], axis=1))
    
    #create future demand (daily) column and ignore rows where Days Prior = 0
    trainingDF['Future Demand(Daily)'] = trainingDF['Future Demand(Total)'] - trainingDF['cum_bookings']
    trainingDF = trainingDF[trainingDF['Days Prior'] != 0]
    
    #create booking rate for multiplicative model
    trainingDF['booking_rate'] = trainingDF['cum_bookings']/trainingDF['Future Demand(Total)']
        
    #finding median using group by function (used because data skewness). also grouped by day of week.
    AdditiveDemandModel = trainingDF.groupby(['Days Prior','Day of Week'])[['Future Demand(Daily)']].median().reset_index()
    multiplyDemandModel = trainingDF.groupby(['Days Prior','Day of Week'])[['booking_rate']].median().reset_index()
    mergeModel = pd.merge(AdditiveDemandModel, multiplyDemandModel, on=['Days Prior', 'Day of Week'], how="left")
    
    #create days prior and day of week column in validation dataset
    validationDF['booking_date'] = pd.to_datetime(validationDF['booking_date'])
    validationDF['departure_date'] = pd.to_datetime(validationDF['departure_date'])
    validationDF['Days Prior'] = (validationDF['departure_date'] - validationDF['booking_date']).dt.days
    validationDF['Day of Week'] = validationDF['departure_date'].dt.weekday_name
    
    #merged validation dataset and our model
    mergedDF = pd.merge(validationDF, mergeModel, on=['Days Prior', 'Day of Week'], how='left')
    mergedDF = mergedDF.dropna(how='any', axis=0) #drop NA values 
    mergedDF.rename(columns={'Future Demand(Daily)':'Average Remaining Demand'}, inplace=True)
    mergedDF['Additive Model'] = mergedDF['Average Remaining Demand'] + mergedDF['cum_bookings'] #create additive model
    mergedDF['Multiplicative Model'] = mergedDF['cum_bookings'] / mergedDF['booking_rate'] #create multiplicative model
    
    #calculate MASE numerator and MASE denominator
    mergedDF['MASE Naive'] = abs(mergedDF['naive_forecast'] - mergedDF['final_demand'])
    maseDenom = mergedDF['MASE Naive'].sum() #find sum of MASE for naive model
    mergedDF['MASE A.M.'] = abs(mergedDF['Additive Model'] - mergedDF['final_demand'])
    maseNumAdditive = mergedDF['MASE A.M.'].sum() #find sum of MASE for additive model
    mergedDF['MASE M.M'] = abs(mergedDF['Multiplicative Model'] - mergedDF['final_demand'])
    maseNumMultiply = mergedDF['MASE M.M'].sum() #find sum of MASE for multiplicative model
    
    #create combined model. uses the row value where it deviates LESS from final demand
    diff = abs(mergedDF['Additive Model'] - mergedDF['final_demand'])
    diff1  = abs(mergedDF['Multiplicative Model'] - mergedDF['final_demand'])
    mergedDF['Combined Model'] = np.where(diff > diff1, mergedDF['Multiplicative Model'], mergedDF['Additive Model'])
    mergedDF['MASE Combined'] = abs(mergedDF['Combined Model'] - mergedDF['final_demand'])
    maseNumCombined = mergedDF['MASE Combined'].sum() #find sum of MASE for combined model
    
    #selecting departure_date, booking_date and A.M. forecasts from model
    displayDF = mergedDF[['departure_date', 'booking_date', 'Combined Model']]
    
    #calculating total MASE
    maseCombinedTotal = "Percentage: {}%".format(float(maseNumCombined/maseDenom)*100)
    
    #create dictionary to return
    finalDict = {'MASE': maseCombinedTotal, 'Forecast Dataframe': displayDF}
    

    print(finalDict)
    print(maseCombinedTotal)
    
airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')

