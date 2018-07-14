library(dplyr)
library(ggplot2)
library(caret)
library(parallel)
library(xgboost)
library(Matrix)
library(onehot)
options(mc.cores = detectCores())

########################################################################
######################## Read and explore the data #####################
## Read Data
baseData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/train.csv"
                    , na.strings = c("NA","","na", " "), stringsAsFactors = FALSE)
givenTestData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/prediction.csv"
                         , na.strings = c("NA","","na", " "), stringsAsFactors = FALSE)
givenTestData$index = seq(1,nrow(givenTestData),1)
baseDataBackup = baseData

#### Explore the data ####
baseData = baseData %>% dplyr :: mutate(rpc = ifelse(Clicks == 0, 0, round(Revenue/Clicks,10))) %>% as.data.frame()
str(baseData)
head(baseData,5)

## Date
dateDistinct = baseData %>% dplyr :: select(Date) %>% distinct() %>% as.data.frame()
min(dateDistinct$Date) # "2014-12-14"
max(dateDistinct$Date) # "2015-04-07"
head(dateDistinct,10)

## Keyword_ID
keywordDistinct = baseData %>% dplyr :: select(Keyword_ID) %>% distinct() %>% as.data.frame()
str(keywordDistinct) # 487981
head(keywordDistinct,10)

## Ad_group_ID
adGroupDistinct = baseData %>% dplyr :: select(Ad_group_ID) %>% distinct() %>% as.data.frame()
str(adGroupDistinct) # 269480
head(adGroupDistinct,10)

## Campaign_ID
campaignDistinct = baseData %>% dplyr :: select(Campaign_ID) %>% distinct() %>% as.data.frame()
str(campaignDistinct) # 2927
head(campaignDistinct,10)

## Account_ID
accountDistinct = baseData %>% dplyr :: select(Account_ID) %>% distinct() %>% as.data.frame()
str(accountDistinct) # 16
head(accountDistinct,20)

# Acount_ID level metrics i.e. revDaily(daily revenue per click), totalClicks(total cicks), totalRevenue(total revenue)
accountRev = baseData %>% dplyr :: group_by(Account_ID, Date) %>%
  dplyr :: summarise(revDaily = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(accountRev)

ggplot(data = accountRev,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~Account_ID) # 575525143937 has outlier

ggplot(data = accountRev,aes(as.Date(Date), totalClicks)) +
  geom_line() + facet_grid(Account_ID~., scale = "free_y")

## Device_ID
deviceDistinct = baseData %>% dplyr :: select(Device_ID) %>% distinct() %>% as.data.frame()
print(deviceDistinct) # 3
head(deviceDistinct)
baseData %>% dplyr :: group_by(Device_ID) %>% dplyr :: summarise(count = n()) %>% 
  as.data.frame() %>% View()

## Match_type_ID
matchTypeDistinct = baseData %>% dplyr :: select(Match_type_ID) %>% distinct() %>% as.data.frame()
str(matchTypeDistinct) # 3
head(matchTypeDistinct)

## Revenue
class(baseData$Revenue)
revenue = baseData %>% dplyr :: select(Revenue) %>% as.data.frame()
boxplot(revenue$Revenue)
head(revenue$Revenue,50)

# Plot the revenue
revenuePerDay = baseData %>% dplyr :: group_by(Date) %>% dplyr :: summarise(revenueDay = sum(Revenue)) %>%
  dplyr :: select(Date, revenueDay) %>% as.data.frame()
str(revenuePerDay)

ggplot(data = revenuePerDay,aes(as.Date(Date), revenueDay)) +
  geom_line() # there's an outlier in the data on 17th Jan, value = 36805267 w.r.t total revenue
boxplot(revenuePerDay$revenueDay) # visible at revenue level

# Check whether the its a real outlier i.e. if the clicks are aligned
rpcDaily = baseData %>% dplyr :: group_by(Date) %>% dplyr :: summarise(revenueDay = sum(Revenue)
                                                                       , rpcDay = sum(revenue)/sum(Clicks)) %>%
  dplyr :: select(Date, rpcDay) %>% as.data.frame()
str(rpcDaily)

ggplot(data = rpcDaily,aes(as.Date(Date), rpcDay)) +
  geom_line() # same for click, 17th Jan 1666.764 is an outlier

# Clicks should also be high on that particular day for sure
clickDaily = baseData %>% dplyr :: group_by(Date) %>% dplyr :: summarise(clickDay = sum(Clicks)) %>%
  dplyr :: select(Date, clickDay) %>% as.data.frame()
str(clickDaily)

ggplot(data = clickDaily,aes(as.Date(Date), clickDay)) +
  geom_line() # Clicks also has the same outlier; value = 885881

## Clicks
unique(baseData$Clicks)
# Click by count
baseData %>% group_by(Clicks) %>% summarise(count = n()) %>% View()
# Daily level clicks aggregation
clicksDaily = baseData %>% dplyr :: group_by(Date) %>% dplyr :: summarise(clickDay = sum(Clicks)) %>%
  dplyr :: select(Date, clickDay) %>% as.data.frame() # same code as before
boxplot(clicksDaily$clickDay)

## Conversions
conversionsDaily = baseData %>% dplyr :: group_by(Date) %>% dplyr :: summarise(conversionDay = sum(Conversions)
                                                                               , convPerClick = sum(Clicks)/sum(Conversions)
                                                                               , totalClicks = sum(Clicks)) %>%
  dplyr :: select(Date, conversionDay, convPerClick, totalClicks) %>% as.data.frame()

ggplot(data = conversionsDaily,aes(as.Date(Date), conversionDay)) +
  geom_line() # Conversions is aligned with the outlier; conversion per click is somewhat normal

# Conversion vs Cliks
ggplot(data = conversionsDaily,aes(as.Date(Date))) +
  geom_line(aes(y=conversionDay), colour = "red") + 
  geom_line(aes(y=convPerClick), colour = "green") + 
  geom_line(aes(y=totalClicks), colour = "blue")

# No conversion case
noConversion = baseData %>% dplyr :: mutate(conv = case_when(.$Conversions>0 ~1, .$Conversions == 0 ~0)) %>% 
  dplyr :: group_by(Date, conv) %>% dplyr :: summarise(conversionDay = sum(Conversions)
                                                       , convPerClick = sum(Clicks)/sum(Conversions)
                                                       , totalClicks = sum(Clicks)) %>%
  dplyr :: select(Date, conv, conversionDay, convPerClick, totalClicks) %>% as.data.frame()
noConversion %>% filter(conv==0) %>% View()

baseData %>% dplyr :: mutate(conv = case_when(.$Conversions>0 ~1, .$Conversions == 0 ~0)) %>%
  dplyr :: group_by(conv) %>% summarise(count = n()) %>% View() # data is too sparse
# conv count
# 0 8123396
# 1 162027

## Keyword + Match type for price setting
# How the different match type influence the revenue per click over time

matchTypePerformance = baseData %>%  dplyr :: group_by(Date, Match_type_ID) %>%
  dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(matchTypePerformance)

ggplot(data = matchTypePerformance,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~ Match_type_ID) # 95725474456, 894413617560 has significant outlier

# 95725474456 -- outlier; highest rpc; maybe the synonym matching
# 872544605608 -- small utlier; decreasing trend
# 894413617560 -- medium outlier

## How the device type influences the revenue

deviceTypePreference = baseData %>% dplyr :: group_by(Date, Device_ID) %>%
  dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(deviceTypePreference)
unique(deviceTypePreference$Device_ID)

ggplot(data = deviceTypePreference,aes(as.Date(Date), totalClicks)) +
  geom_line() + facet_grid(~ Device_ID) # 298643508640 has significant outlier

format(unique(baseData$Device_ID), scientific = FALSE) 
# 298643508640 - highest rpc
# 848779586902 - lowest rpc
# 1077718730738 - medium; has an outlier

# Relation between Device and match type
deviceVsmatch = baseData %>% dplyr :: filter(Match_type_ID == 894413617560) %>% 
  dplyr :: group_by(Date, Device_ID) %>% dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)
                                                            , totalClicks = sum(Clicks)
                                                            , totalRevenue = sum(Revenue)) %>%
  as.data.frame()
head(deviceVsmatch)

ggplot(data = deviceVsmatch,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~ Device_ID) # particular match type is run on all devices

# Check the reverse
matchVsdevice = baseData %>% dplyr :: filter(Device_ID == 1077718730738) %>% 
  dplyr :: group_by(Date, Match_type_ID) %>% dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)) %>%
  as.data.frame()
head(matchVsdevice)

ggplot(data = matchVsdevice,aes(as.Date(Date), rpcAvg)) +
  geom_line() + facet_grid(~ Match_type_ID) # so basically for all devices all kinds of keyword match type is run

## Explore the Keyword grouping
# Keyword + adgroup is unique
# account <-- campaign <-- adgroup <-- keywords (<-- matchtypeid)

# Relation between account and device id
unique(baseData$Account_ID)

baseData %>% dplyr :: group_by(Account_ID, Device_ID) %>%
  dplyr :: summarise(count = n()) %>% as.data.frame() %>% View() # no specific




######################################################################
########################## Feature Engineering #######################

#### Dummy variable creation ####

## 2 dummies for 3 categories of Match_type_ID
#	95725474456 : 3754784; has highest rpc, totalRevenue and totalClicks
# 872544605608 : 1415356 
# 894413617560 : 3115283

baseData = baseData %>% dplyr :: mutate(matchType1 = case_when(.$Match_type_ID == 95725474456 ~ 1
                                                               , .$Match_type_ID != 95725474456 ~ 0)) %>% as.data.frame()
baseData = baseData %>% dplyr :: mutate(matchType2 = case_when(.$Match_type_ID == 894413617560 ~ 1
                                                               , .$Match_type_ID != 894413617560 ~ 0)) %>% as.data.frame()

## 2 dummies for 3 Device_ID's
# 298643508640 : 3980401
# 848779586902 : 2754187
# 1077718730738 : 1550835
baseData = baseData %>% dplyr :: mutate(device1 = case_when(.$Device_ID == 298643508640 ~ 1
                                                            , .$Device_ID != 298643508640 ~ 0)) %>% as.data.frame()
baseData = baseData %>% dplyr :: mutate(device2 = case_when(.$Device_ID == 848779586902 ~ 1
                                                            , .$Device_ID != 848779586902 ~ 0)) %>% as.data.frame()

## Conversion per click : cpc
baseData = baseData %>% 
  dplyr :: mutate(cpc = ifelse(Clicks == 0,0,round(Conversions/Clicks,10))) %>% as.data.frame()

#### outlier removal ####
# Device_ID = 298643508640 has highest and rest has medium outlier
# Match_type_ID = all of them; 872544605608 is comparatively less significant
# Account_ID = 575525143937 has the outlier for date = '2015-01-17'

countAccount = baseData %>% dplyr :: group_by(Date, Account_ID) %>%
  dplyr :: summarise(count = n()) %>% as.data.frame()
head(countAccount)

ggplot(data = countAccount,aes(as.Date(Date), count)) +
  geom_line() + facet_grid(~Account_ID) 

barForClicks = baseData %>% dplyr :: filter(Date == '2015-01-17' & Account_ID == 575525143937) %>%
  group_by(Clicks) %>% summarise(count = n()) %>% as.data.frame()

# We need bar plot to check whether the clicks=85 count is an outlier
ggplot(data = barForClicks,aes(Clicks, count)) +
  geom_bar(stat = "identity") 

# We remove the data for "2015-01-17" for Account_ID == 575525143937 as an outlier removal 
baseDataOutlier = baseData %>% dplyr :: filter(Date == '2015-01-17' & Account_ID == 575525143937) %>%
  as.data.frame() 
str(baseDataOutlier) # 215883

# Keyword_ID + Ad_group_ID is unique
baseDataOutlierRemoved = baseData %>% anti_join(baseDataOutlier
                                                , by = c("Keyword_ID", "Ad_group_ID", "Date", "Account_ID")) %>%
  as.data.frame()
str(baseDataOutlierRemoved) #8069540
str(baseData)# 8285423
215883 + 8069540

## Check if outlier is removed 
# Device_ID level
deviceTypePreferenceCheck = baseDataOutlierRemoved %>% dplyr :: group_by(Date, Device_ID) %>%
  dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(deviceTypePreferenceCheck)
unique(deviceTypePreferenceCheck$Device_ID)

ggplot(data = deviceTypePreferenceCheck,aes(as.Date(Date), totalClicks)) +
  geom_line() + facet_grid(~ Device_ID) # outlier removed at device_id level

# Account_ID level
accountRevCheck = baseDataOutlierRemoved %>% dplyr :: group_by(Account_ID, Date) %>%
  dplyr :: summarise(revDaily = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(accountRevCheck)

ggplot(data = accountRevCheck,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~Account_ID) # outlier removed at account_id level

ggplot(data = accountRevCheck,aes(as.Date(Date), totalClicks)) +
  geom_line() + facet_grid(Account_ID~., scale = "free_y")

# Match type level
matchTypeCheck = baseDataOutlierRemoved %>%  dplyr :: group_by(Date, Match_type_ID) %>%
  dplyr :: summarise(rpcAvg = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(matchTypeCheck)

ggplot(data = matchTypeCheck,aes(as.Date(Date), rpcAvg)) +
  geom_line() + facet_grid(~ Match_type_ID) # outlier is gone




#################################################################
########################### Model Fit ###########################

head(baseDataOutlierRemoved)
# First model using only Match_Type, device, cpc
fullData = baseDataOutlierRemoved %>% dplyr :: select(Date
                                                      , Clicks
                                                      , rpc
                                                      , matchType1
                                                      , matchType2
                                                      , device1
                                                      , device2
                                                      , cpc) %>% as.data.frame()
head(fullData)
str(fullData)

#### Test train split ####
set.seed(123)

intrain = caret :: createDataPartition(y=fullData$rpc,p=0.7,list=FALSE)
trainData = fullData[intrain,]
testData = fullData[-intrain,]
str(trainData)
str(testData)
2420862 + 5648678
colnames(fullData)

#### LM Model ####

lmModel = lm(rpc ~ matchType1 + matchType2 + device1 + device2
             , data = trainData)
summary(lmModel)
str(lmModel)

mseTrainLM = mean(lmModel$residuals^2)
rmseTrainLM = sqrt(mseTrainLM) 
print(rmseTrainLM) # 725.0209

testDataLM = testData %>% dplyr :: select(matchType1, matchType2, device1, device2) %>% as.data.frame()
testPredictLM = predict(lmModel, testDataLM)
testPredictFrameLM = data.frame("residuals" = testPredictLM - testData$rpc
                                , "testPredict" = testPredictLM
                                , "testReal" = testData$rpc
                                , "Date" = testData$Date)
testRMSELM = sqrt(mean(testPredictFrameLM$residuals^2))
print(testRMSELM) # 705.2692
# test and train rmse are closer; the model is not overfitting but both are high bias

testPredictFramePlot = testPredictFrameLM %>% group_by(Date) %>% summarise(predictRPC = sum(testPredict)
                                                                           , realRPC = sum(testReal)) %>% as.data.frame()

ggplot(data = testPredictFramePlot, aes(as.Date(Date))) + 
  geom_line(aes(y = predictRPC), colour = "red") +
  geom_line(aes(y = realRPC), colour = "green")

#### LM Model with Interactin ####
lmModelInteraction = lm(rpc ~ matchType1 + matchType2 + device1 + device2 +
                          #matchType1:matchType2 + device1:device2 + 
                          matchType1:device1 + 
                          matchType1:device2 + matchType2:device1 + matchType2:device2
                        , data = trainData)
summary(lmModelInteraction)
str(lmModelInteraction)

mseLMInteraction = mean(lmModelInteraction$residuals^2)
rmseTrainLMInteraction = sqrt(mseLMInteraction) 
print(rmseTrainLMInteraction) # 725.0105

testDataLMI = testData %>% dplyr :: select(matchType1, matchType2, device1, device2)
testPredictLMI = predict(lmModelInteraction, testDataLMI)
testPredictFrameLMI = data.frame("residuals" = testPredictLMI - testData$rpc
                                 , "testPredict" = testPredictLMI
                                 , "testReal" = testData$rpc
                                 , "Date" = testData$Date)
testRMSELMI = sqrt(mean(testPredictFrameLMI$residuals^2)) 
print(testRMSELMI)# 705.2612
testPredictFrameLMI %>% filter(testReal != 0) %>% View()
# again test and train rmse are closer; the model is not overfitting but both are high bias

testPredictFramePlotLMI = testPredictFrameLMI %>% group_by(Date) %>% summarise(predictRPC = sum(testPredict)
                                                                               , realRPC = sum(testReal)) %>% as.data.frame()

ggplot(data = testPredictFramePlotLMI, aes(as.Date(Date))) + 
  geom_line(aes(y = predictRPC), colour = "red") +
  geom_line(aes(y = realRPC), colour = "green")

## From simple LM to LM with interaction gives negligible improvement only in train rmse

#### LM with Cross validation #####
## CV with 10 folds using caret package

nFolds = 10
folds = rep_len(1:nFolds, nrow(fullData))
folds = sample(folds, nrow(fullData))

trainRMSELMCV = vector()
testRMSELMCV = vector()

# Run the model in a loop with different folds
for(k in 1:nFolds) {
  
  fold = which(folds == k)
  trainDataLMCV = fullData[-fold,]
  testDataLMCV = fullData[fold,]
  
  lmModelCV = lm(rpc ~ matchType1 + matchType2 + device1 + device2
                 , data = trainDataLMCV)
  
  mseLMCV = mean(lmModelCV$residuals^2)
  rmseTrainLMCV = sqrt(mseLMCV) 
  trainRMSELMCV = c(trainRMSELMCV, rmseTrainLMCV)
  
  testPredictLMCV = predict(lmModelCV, testDataLMCV)
  
  rmseTestLMCV = sqrt(mean((testPredictLMCV - testDataLMCV$rpc)^2))
  testRMSELMCV = c(testRMSELMCV, rmseTestLMCV)
  
}

indexLMCV = seq(1,length(trainRMSELMCV),1)

rmseLMCV = data.frame("indexLMCV" = indexLMCV
                      , "trainRMSELMCV" = trainRMSELMCV
                      , "testRMSELMCV" = testRMSELMCV)

ggplot(data = rmseLMCV, aes(indexLMCV)) +
  geom_line(aes(y = trainRMSELMCV), colour = "blue") + 
  geom_line(aes(y = testRMSELMCV), colour = "red")

mean(rmseLMCV$trainRMSELMCV) # 719.1056
mean(rmseLMCV$testRMSELMCV) # 715.9512

## Even with the CV the model doesn't seem to improve; need to go to non-linearity with different models
## Basically the model is a naive mean predictor model; for every specific combination this is giving me the mean rpc




#### Create new feature and go into the space of non-linearity ####
## make groups for the accounts; 5 dummy variables ##

accountRevCheck = baseDataOutlierRemoved %>% dplyr :: group_by(Account_ID, Date) %>%
  dplyr :: summarise(revDaily = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
head(accountRevCheck)

ggplot(data = accountRevCheck,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~Account_ID) # outlier removed at account_id level

# Account_ID
# 1  861287123742 mid
# 2  654870334100 high - mid
# 3  212779990172 high
# 4  575525143937 mid - low
# 5  719583196582 high - mid
# 6  221354172146 mid
# 7  151664859558 mid - low
# 8  604905316813 high
# 9  256188843610 low
# 10 412971074791 low
# 11 981453654147 mid - low
# 12 573604300663 low
# 13 602182847798 low
# 14 341124366337 low
# 15 866124423689 low
# 16 164144662657 low
baseDataOutlierRemoved = baseDataOutlierRemoved %>% dplyr :: mutate(accountIndicator = case_when(.$Account_ID == 212779990172 ~ 5
                                                                                                 , .$Account_ID == 604905316813 ~ 5
                                                                                                 , .$Account_ID == 654870334100 ~ 2.5
                                                                                                 , .$Account_ID == 719583196582 ~ 2.5
                                                                                                 , .$Account_ID == 221354172146 ~ 2
                                                                                                 , .$Account_ID == 861287123742 ~ 2
                                                                                                 , .$Account_ID == 575525143937 ~ 1.25
                                                                                                 , .$Account_ID == 981453654147 ~ 1.25
                                                                                                 , .$Account_ID == 151664859558 ~ 1.25
                                                                                                 , .$Account_ID == 256188843610 ~ 1
                                                                                                 , .$Account_ID == 412971074791 ~ 1
                                                                                                 , .$Account_ID == 573604300663 ~ 1
                                                                                                 , .$Account_ID == 602182847798 ~ 1
                                                                                                 , .$Account_ID == 341124366337 ~ 1
                                                                                                 , .$Account_ID == 866124423689 ~ 1
                                                                                                 , .$Account_ID == 164144662657 ~ 1)) %>% as.data.frame()
baseDataOutlierRemoved = baseDataOutlierRemoved %>% filter(rpc <= 98636.21) %>% as.data.frame()
baseDataOutlierRemoved = baseDataOutlierRemoved %>% mutate(rpcBinary = case_when(.$rpc>0 ~1
                                                                                 , .$rpc==0 ~0)) %>%
  as.data.frame()
write.csv(baseDataOutlierRemoved, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/baseDataOutlierRemoved.csv")
baseDataOutlierRemoved = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/baseDataOutlierRemoved.csv")
  
head(baseDataOutlierRemoved)
sds = baseDataOutlierRemoved %>% dplyr :: group_by(accountIndicator, Date) %>%
  dplyr :: summarise(revDaily = sum(Revenue)/sum(Clicks)
                     , totalClicks = sum(Clicks)
                     , totalRevenue = sum(Revenue)) %>% as.data.frame()
ggplot(data = sds,aes(as.Date(Date), totalRevenue)) +
  geom_line() + facet_grid(~accountIndicator)

fullDataAccountID = baseDataOutlierRemoved %>% dplyr :: select(Date, Revenue, Clicks, Conversions
                                                               , rpc, matchType1, matchType2, device1
                                                               , device2, cpc, accountIndicator)
head(fullDataAccountID)
str(fullDataAccountID)

#### LM with polynomial feature ####

nFolds = 10
foldsLMAccount = rep_len(1:nFolds, nrow(fullDataAccountID))
foldsLMAccount = sample(foldsLMA, nrow(fullDataAccountID))

trainRMSELMAccountCV = vector()
testRMSELMAccountCV = vector()

# Run the model in a loop with different folds
for(k in 1:nFolds) {
  
  fold = which(foldsLMAccount == k)
  trainDataLMAccountCV = fullDataAccountID[-fold,]
  testDataLMAccountCV = fullDataAccountID[fold,]
  
  lmModelAccountCV = lm(rpc ~ matchType1 + matchType2 + device1 + device2 + accountIndicator
                        + I(accountIndicator^2) + I(accountIndicator^3)
                        , data = trainDataLMAccountCV)
  
  mseLMAccountCV = mean(lmModelAccountCV$residuals^2)
  rmseTrainLMAccountCV = sqrt(mseLMAccountCV) 
  trainRMSELMAccountCV = c(trainRMSELMAccountCV, rmseTrainLMCV)
  
  testPredictLMCV = predict(lmModelAccountCV, testDataLMAccountCV)
  
  rmseTestLMAccountCV = sqrt(mean((testPredictLMCV - testDataLMAccountCV$rpc)^2))
  testRMSELMAccountCV = c(testRMSELMAccountCV, rmseTestLMAccountCV)
  
}

indexLMCV = seq(1,length(trainRMSELMAccountCV),1)

rmseLMAccountCV = data.frame("indexLMCV" = indexLMCV
                             , "trainRMSELMCV" = trainRMSELMAccountCV
                             , "testRMSELMCV" = testRMSELMAccountCV)

ggplot(data = rmseLMAccountCV, aes(indexLMCV)) +
  geom_line(aes(y = trainRMSELMAccountCV), colour = "blue") + 
  geom_line(aes(y = testRMSELMAccountCV), colour = "red")

mean(rmseLMAccountCV$trainRMSELMCV) # 715.7639
mean(rmseLMAccountCV$testRMSELMCV) # 714.4408

realVsPredictLMAccount = data.frame("real" = testDataLMAccountCV$rpc
                                    , "predict" = testPredictLMCV)
head(realVsPredictLMAccount,100)

## Only a slight improvement; still the naive mean prediction; polynomial terms didn't help much



#### Two Step Approach : 1. Classify if rpc>0 then find rpc value ####
#### XGBoost to apply only on rpc > 0 ####
set.seed(1234)

fullDataXGB = baseDataOutlierRemoved %>% dplyr :: select(Date, rpc, matchType1, matchType2
                                                         , device1, device2, accountIndicator)

# only positive rpc
fullDataXGB = fullDataXGB %>% filter(rpc > 0) %>% as.data.frame()
fullDataXGB$logRPC = log(fullDataXGB$rpc)
hist(log(fullDataXGB$rpc))

intrain = caret :: createDataPartition(y=fullDataXGB$logRPC,p=0.8,list=FALSE)
trainData = fullDataXGB[intrain,]
testData = fullDataXGB[-intrain,]
testData$accountIndicator2 = testData$accountIndicator^2
testData$accountIndicator3 = testData$accountIndicator^3
testData$accountIndicator4 = testData$accountIndicator^4

xgbTrain = trainData[, !(colnames(trainData) %in% c("rpc", "Date", "logRPC"))]
head(xgbTrain)
nrow(xgbTrain)
xgbTrain$accountIndicator2 = xgbTrain$accountIndicator^2
xgbTrain$accountIndicator3 = xgbTrain$accountIndicator^3
xgbTrain$accountIndicator4 = xgbTrain$accountIndicator^4

xgbTrainLabel = trainData[, (colnames(trainData) %in% c("logRPC")), drop = FALSE]
head(xgbTrainLabel,15)
nrow(xgbTrainLabel)

# Model XGBoost
xgbTree = xgboost(data = as.matrix(xgbTrain)
                  , label = xgbTrainLabel$logRPC
                  , max_depth = 10
                  , eta = .3
                  #, gamma = .000001
                  , nthread = 5
                  , nrounds = 100
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .8
                  , colsample_bytree = .8
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

predXGB = predict(xgbTree, newdata = as.matrix(testData[, c(3,4,5,6,7,9,10,11)]))
testRMSEXGB = sqrt(mean((testData$logRPC - predXGB)^2))
print(testRMSEXGB) # 656.409729 vs 653.8248 

xgbAnalyse = data.frame("real" = testData$rpc
                        , "predicted" = exp(predXGB)
                        , "Date" = testData$Date)

xgbAnalyse %>% dplyr :: filter(real>0) %>% View()
xgbAnalyse %>% dplyr :: filter(real==0) %>% View()

xgbAnalyseDaily = xgbAnalyse %>% dplyr :: group_by(Date) %>% summarise("realVal" = sum(real)
                                                                       , "predictVal" = sum(predicted)) %>%
  as.data.frame()

ggplot(data = xgbAnalyseDaily, aes(as.Date(Date))) +
  geom_line(aes(y = realVal), colour = "blue") +
  geom_line(aes(y = predictVal), colour = "red")




#### Dive into the campaign level matrix ####
histRPC = baseDataOutlierRemoved %>% filter(rpc>0 & rpc <= 98636.21) %>% select(rpc) %>% View()
hist(histRPC$rpc)

# Only Keyword_ID + Ad_group_ID with positive rpc
keywordADLevel = baseDataOutlierRemoved %>% group_by(Keyword_ID, Ad_group_ID) %>% 
  summarise(count=n()
            , rpcTotal = sum(Revenue)
            , clicks = sum(Clicks)
            , rpc = sum(Revenue)/sum(Clicks)) %>% filter(rpc>0) %>%
  arrange(desc(rpc)) %>% as.data.frame()
nrow(keywordADLevel) # 69468 keyword + adgroup combination with rpc > 0

# Total distinct combincation of Keyword_ID + Ad_group_ID 
keywordADLevelAll = baseDataOutlierRemoved %>% group_by(Keyword_ID, Ad_group_ID) %>% 
  summarise(count=n()
            , rpcTotal = sum(Revenue)
            , clicks = sum(Clicks)
            , rpc = sum(Revenue)/sum(Clicks)) %>% filter(rpc>=0) %>%
  arrange(desc(rpc)) %>% as.data.frame()
nrow(keywordADLevelAll) # 1051579 total keyword + adgroup combination

# From rpc>0; how many rows of Keyword_ID + Ad_group_ID are there is train data set
keywordADwithRPC = keywordADLevel %>% inner_join(baseDataOutlierRemoved, by = c("Keyword_ID","Ad_group_ID")) %>%
  as.data.frame() 
nrow(keywordADwithRPC) # 2737400

# From rpc>0; how many rows of Keyword_ID + Ad_group_ID are there is given test data set
keywordADwithRPCGivenTest = keywordADLevel %>% inner_join(givenTestData, by = c("Keyword_ID","Ad_group_ID")) %>%
  as.data.frame()
nrow(keywordADwithRPCGivenTest) # 151727
578012 - 151727 = 426285 # has keyword + ad with no rpc in train data

keywordADwithRPC %>% select(Campaign_ID) %>% distinct() %>% View() # 1302

# from all camapigns how many are having rpc>0
campaignLevel = baseDataOutlierRemoved %>% group_by(Campaign_ID) %>% 
  summarise(count=n()
            , rpcTotal = sum(Revenue)
            , clicks = sum(Clicks)
            , rpc = sum(Revenue)/sum(Clicks)) %>% filter(rpc>0) %>%
  arrange(desc(rpc)) %>% as.data.frame()
nrow(campaignLevel) # 1302 (total campaigns=2927)
View(campaignLevel)

baseDataOutlierRemoved %>% filter(rpcBinary == 1) %>% select(Campaign_ID) %>% distinct() %>% View()
baseDataOutlierRemovedIndex %>% filter(rpcBinary == 1) %>% select(Campaign_ID) %>% distinct() %>% View()

campaignWithRPC = campaignLevel %>% inner_join(baseDataOutlierRemoved, by = c("Campaign_ID")) %>%
  as.data.frame()
nrow(campaignWithRPC) # 7979791
campaignWithRPCGivenTest = campaignLevel %>% inner_join(givenTestData, by = c("Campaign_ID")) %>%
  as.data.frame()
nrow(campaignWithRPCGivenTest) # 570628
nrow(givenTestData) # 578012
578012 - 570628 = 7384 # has campaign with zero rpc in test data
nrow(baseDataOutlierRemoved) # 8069519

# Change the given test data set with indexed value according to keyword+ad_account
givenTestData1 = givenTestData %>% 
  left_join(keywordADwithRPCGivenTest[,11:12, drop = FALSE], by = c("index"), suffix = c(".x", ".y")) %>%
  as.data.frame()
names = colnames(givenTestData)
colnames(givenTestData1) = c(names, "index2")

givenTestData1 = givenTestData1 %>% 
  mutate(indexBinary = case_when(is.na(.$index2) ~ 0
                                 , !is.na(.$index2) ~ 1)) %>% as.data.frame()

givenTestData1 = givenTestData1 %>% select(-index2) %>% as.data.frame()
givenTestData = givenTestData1

# Change the baseDataOutlierRemoved according to keyword + ad_account
keywordADLevelJoin = keywordADLevel %>% dplyr :: select(Keyword_ID, Ad_group_ID) %>% as.data.frame()
keywordADLevelJoin$indexBinary = 1
baseDataOutlierRemovedIndex = baseDataOutlierRemoved %>% dplyr :: left_join(keywordADLevelJoin, suffixes = c(".x", ".y")) %>%
  replace(is.na(.),0) %>% as.data.frame() 

baseDataOutlierRemovedIndex %>% group_by(indexBinary) %>% summarise(count = n()) %>%
  View() # has 2737400 non zero values

# This is the new test data set
write.csv(givenTestData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/predictionIndexed.csv")
write.csv(keywordADLevel, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/keywordADLevel.csv")
write.csv(campaignLevel, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/campaignLevel.csv")
write.csv(baseDataOutlierRemovedIndex, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/baseDataOutlierRemovedIndex.csv")



#########################################################################
###################### Two Step Approach Modelling ######################
#### Step 1 : Classification whether a combination will get revenue #####

baseDataOutlierRemovedIndex = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/baseDataOutlierRemovedIndex.csv")
head(baseDataOutlierRemovedIndex)
baseDataOutlierRemovedIndex %>% group_by(indexBinary, rpcBinary) %>% summarise(count = n()) %>%
  View()

# Data only with considered keyword and ad_group_id

twoStepData = baseDataOutlierRemovedIndex %>% dplyr :: filter(indexBinary == 1) %>% 
  as.data.frame()

twoStepData %>% filter(rpc>0 & rpc <= 98636.21) %>% group_by(Campaign_ID) %>% summarise(revTotal = sum(Revenue)
                                                    , clickTotal = sum(Clicks)
                                                    , rpc = sum(Revenue)/sum(Clicks)
                                                    , count = n()) %>%
  as.data.frame() %>% View()

twoStepData = twoStepData %>% dplyr :: select(Date, Keyword_ID, Ad_group_ID, Campaign_ID
                                              , Account_ID, Device_ID, Match_type_ID
                                              , Revenue, Clicks, Conversions, rpc
                                              , rpcBinary, accountIndicator, indexBinary)

# one-hot for Match_type_ID
twoStepDataTemp = twoStepData
twoStepDataTemp$Match_type_ID = factor(twoStepData$Match_type_ID)
encoder = onehot(twoStepDataTemp[,c("Match_type_ID"), drop = FALSE])
oneHotPredict = predict(encoder, twoStepDataTemp[,c("Match_type_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
View(twoStepData)

# one-hot for Device_ID
twoStepDataTemp$Device_ID = factor(twoStepData$Device_ID)
encoder = onehot(twoStepDataTemp[,c("Device_ID"), drop = FALSE])
oneHotPredict = predict(encoder, twoStepDataTemp[,c("Device_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
View(twoStepData)

# one-hot for Account_ID
twoStepDataTemp$Account_ID = factor(twoStepData$Account_ID)
encoder = onehot(twoStepDataTemp[,c("Account_ID"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataTemp[,c("Account_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
View(twoStepData)

write.csv(twoStepData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataOneHotMatchDeviceAccount.csv")
twoStepData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataOneHotMatchDeviceAccount.csv")
head(twoStepData)
nrow(twoStepData)

# this data is used for all
twoStepDataPython = twoStepData %>% select(-c(X
                                              , Date
                                              , Keyword_ID
                                              , Ad_group_ID
                                              , Account_ID
                                              , Device_ID
                                              , Match_type_ID
                                              , Revenue
                                              , Clicks
                                              , Conversions
                                              , rpc
                                              , accountIndicator
                                              , indexBinary
                                              )) %>% as.data.frame()
twoStepDataPython %>% group_by(rpcBinary) %>% summarise(count = n()) %>% View()

write.csv(twoStepDataPython, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataPython.csv", row.names=FALSE)

twoStepDataPython %>% filter(rpcBinary==1) %>% select(Campaign_ID) %>% distinct() %>% View()
twoStepData %>% filter(rpcBinary == 1) %>% select(Campaign_ID) %>% distinct() %>% View()

twoStepData %>% filter(indexBinary==1) %>% group_by(rpcBinary) %>% summarise(count = n()) %>%
  as.data.frame() %>% View()
0.05750018
0.9424998

# one-hot for Campaign_ID
twoStepDataTemp = twoStepData

twoStepData %>% group_by(Campaign_ID) %>% summarise(rpc1 = sum(Revenue)/sum(Clicks)
                                                    , click = sum(Clicks)
                                                    ) %>%
  mutate(campaignIndex = case_when(.$rpc1>=5000~ 100
                                   , .$rpc1>=2500 & .$rpc1 <5000 ~ 80
                                   , .$rpc1>=1200 & .$rpc1 <2500 ~ 60
                                   , .$rpc1>=800 & .$rpc1 <1200 ~ 40
                                   , .$rpc1>=500 & .$rpc1 <800 ~ 20
                                   , .$rpc1>=300 & .$rpc1 <500 ~ 10
                                   , .$rpc1>=200 & .$rpc1 <300 ~ 8
                                   , .$rpc1>=100 & .$rpc1 <200 ~ 6
                                   , .$rpc1>=50 & .$rpc1 <100 ~ 4
                                   , .$rpc1>0 & .$rpc1 <50 ~ 1
                                   , .$rpc1==0 ~ 0)) %>% 
  group_by(campaignIndex) %>% summarise(count = n()) %>% as.data.frame() %>% View()

campaignIndex = twoStepData %>% group_by(Campaign_ID) %>% summarise(rpc1 = sum(Revenue)/sum(Clicks)
                                                    , click = sum(Clicks)
) %>%
  mutate(campaignIndex = case_when(.$rpc1>=5000~ 100
                                   , .$rpc1>=2500 & .$rpc1 <5000 ~ 80
                                   , .$rpc1>=1200 & .$rpc1 <2500 ~ 60
                                   , .$rpc1>=800 & .$rpc1 <1200 ~ 40
                                   , .$rpc1>=500 & .$rpc1 <800 ~ 20
                                   , .$rpc1>=300 & .$rpc1 <500 ~ 10
                                   , .$rpc1>=200 & .$rpc1 <300 ~ 8
                                   , .$rpc1>=100 & .$rpc1 <200 ~ 6
                                   , .$rpc1>=50 & .$rpc1 <100 ~ 4
                                   , .$rpc1>0 & .$rpc1 <50 ~ 1
                                   , .$rpc1==0 ~ 0)) %>% select(Campaign_ID,campaignIndex) %>% as.data.frame()
View(campaignIndex)

twoStepData = twoStepData %>% left_join(campaignIndex, by = c("Campaign_ID")) %>% as.data.frame()
colnames(twoStepData)

twoStepDataTemp = twoStepData
twoStepDataTemp$campaignIndex = factor(twoStepData$campaignIndex)
encoder = onehot(twoStepDataTemp[,c("campaignIndex"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataTemp[,c("campaignIndex"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
View(twoStepData)

write.csv(twoStepData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepData.csv", row.names=FALSE)

# Date index
dateBinary = twoStepData %>% group_by(Date, rpcBinary) %>% summarise(count1 = n()) %>% as.data.frame()
dateTotal = twoStepData %>% group_by(Date) %>% summarise(count2 = n()) %>% as.data.frame()

dateIndex = dateBinary %>% left_join(dateTotal, by = c("Date")) %>% mutate(dateIndex = count1/count2) %>%
  filter(rpcBinary == 1) %>% select(Date, dateIndex) %>% as.data.frame()

twoStepData = twoStepData %>% left_join(dateIndex, by = c("Date")) %>% as.data.frame()

twoStepDataTemp = twoStepData
twoStepDataTemp$Date = factor(twoStepData$Date)
encoder = onehot(twoStepDataTemp[,c("Date"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataTemp[,c("Date"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
View(twoStepData)


# Keyword + adgroupid index; 
# for classification:conversion rate 
# for regression:rpc

keyADIndexConversion = baseDataOutlierRemoved %>% group_by(Keyword_ID, Ad_group_ID) %>% 
  summarise(count=n()
            , clicks = sum(Clicks)
            , rev = sum(Revenue)
            , cpc = sum(Conversions)/sum(Clicks)) %>% filter(cpc>0) %>% as.data.frame()
keyADIndexConversion %>%  mutate(keyADConversionIndex = case_when(.$cpc<=.01 ~ 1
                                , .$cpc>=.01 & .$cpc<.02 ~ 2
                                , .$cpc>=.02 & .$cpc<.03 ~ 3
                                , .$cpc>=.03 & .$cpc<.04 ~ 4
                                , .$cpc>=.04 & .$cpc<.05 ~ 5
                                , .$cpc>=.05 & .$cpc<.06 ~ 6
                                , .$cpc>=.06 & .$cpc<.07 ~ 7
                                , .$cpc>=.07 & .$cpc<.08 ~ 8
                                , .$cpc>=.08 & .$cpc<.09 ~ 9
                                , .$cpc>=.09 & .$cpc<.1 ~ 10
                                , .$cpc>=.1 & .$cpc<.15 ~ 15
                                , .$cpc>=.15 & .$cpc<.25 ~ 25
                                , .$cpc>=.25 & .$cpc<.5 ~ 50
                                , .$cpc>=.5 & .$cpc<1 ~ 100
                                , .$cpc>=1 & .$cpc<1.25 ~ 125
                                , .$cpc>=1.25 ~ 250
                                )) %>% group_by(keyADConversionIndex) %>% summarise(count = n()) %>% View()

# .01 is the base for this
# .01 till .1 then .1 to .2 etc

keyADIndexConversionOnehot = keyADIndexConversion %>%  
  mutate(keyADConversionIndex = case_when(.$cpc<.01 ~ 1
                                          , .$cpc>=.01 & .$cpc<.02 ~ 2
                                          , .$cpc>=.02 & .$cpc<.03 ~ 3
                                          , .$cpc>=.03 & .$cpc<.04 ~ 4
                                          , .$cpc>=.04 & .$cpc<.05 ~ 5
                                          , .$cpc>=.05 & .$cpc<.06 ~ 6
                                          , .$cpc>=.06 & .$cpc<.07 ~ 7
                                          , .$cpc>=.07 & .$cpc<.08 ~ 8
                                          , .$cpc>=.08 & .$cpc<.09 ~ 9
                                          , .$cpc>=.09 & .$cpc<.1 ~ 10
                                          , .$cpc>=.1 & .$cpc<.15 ~ 15
                                          , .$cpc>=.15 & .$cpc<.25 ~ 25
                                          , .$cpc>=.25 & .$cpc<.5 ~ 50
                                          , .$cpc>=.5 & .$cpc<1 ~ 100
                                          , .$cpc>=1 & .$cpc<1.25 ~ 125
                                          , .$cpc>=1.25 ~ 250
  )) %>%select(Keyword_ID, Ad_group_ID, keyADConversionIndex) %>% as.data.frame()
  
twoStepData = twoStepData %>% left_join(keyADIndexConversionOnehot, by = c("Keyword_ID","Ad_group_ID")) %>%
  as.data.frame()

write.csv(twoStepData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepData.csv", row.names=FALSE)

twoStepDataTemp = twoStepData
twoStepDataTemp$keyADConversionIndex = factor(twoStepData$keyADConversionIndex)
encoder = onehot(twoStepDataTemp[,c("keyADConversionIndex"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataTemp[,c("keyADConversionIndex"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepData)
head(twoStepData)

# Date conversion index
dateIndexConversion = baseDataOutlierRemoved %>% group_by(Date) %>% 
  summarise(count=n()
            , clicks = sum(Clicks)
            , rev = sum(Revenue)
            , dateIndexRevenue = round(sum(Revenue)/sum(Clicks),0)
            , dateIndexConversion = round(10000*sum(Conversions)/sum(Clicks),0)) %>%
  dplyr :: select(Date, dateIndexRevenue, dateIndexConversion) %>% filter(dateIndexConversion>0) %>% as.data.frame()

View(dateIndexConversion)

twoStepData = twoStepData %>% left_join(dateIndexConversion, by = c("Date")) %>%
  as.data.frame()
ncol(twoStepData)
head(twoStepData1)

twoStepDataTemp = twoStepData
twoStepDataTemp$dateIndexConversion = factor(twoStepData$dateIndexConversion)
encoder = onehot(twoStepDataTemp[,c("dateIndexConversion"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataTemp[,c("dateIndexConversion"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepData = twoStepData %>% cbind(dfOneHot) %>% as.data.frame()

nrow(twoStepData)
ncol(twoStepData)
head(twoStepData)

twoStepDataFinal = twoStepData
write.csv(twoStepDataFinal, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataFinal.csv", row.names=FALSE)

grep("Date=" ,names(twoStepDataFinal))

colnames(twoStepDataFinal) %>% dplyr:: select(starts_with("Date=")) %>% View()
grep("Date=",colnames(twoStepDataFinal), value = TRUE) 



twoStepDataFinal = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataFinal.csv")

#### Prepare prediction data for classification and regression

# accountIndicator
accountIndicatorPred = twoStepDataFinal %>% select(Account_ID, accountIndicator) %>% group_by(Account_ID, accountIndicator) %>% 
  summarise(count = n()) %>% select(Account_ID, accountIndicator) %>% as.data.frame()

givenTestData = givenTestData %>% left_join(accountIndicatorPred, by = c("Account_ID")) %>%
  as.data.frame()

# indexBinary
indexBinaryPred = twoStepDataFinal %>% group_by(Keyword_ID, Ad_group_ID) %>% 
  summarise(count = n(), rpcl = sum(Revenue)/sum(Clicks)) %>%  filter(rpcl>0) %>% select(Keyword_ID, Ad_group_ID) %>% as.data.frame()
indexBinaryPred$indexBinary = 1

givenTestData %>% left_join(indexBinaryPred, by = c("Keyword_ID", "Ad_group_ID")) %>%
  replace(is.na(.),0) %>% select(indexBinary) %>% group_by(indexBinary) %>%
  summarise(count = n()) %>% View()

givenTestData = givenTestData %>% left_join(indexBinaryPred, by = c("Keyword_ID", "Ad_group_ID")) %>%
  replace(is.na(.),0) %>% as.data.frame()

# one-hot : Match type ID 
givenTestDataTemp = givenTestData
givenTestDataTemp$Match_type_ID = factor(givenTestData$Match_type_ID)
encoder = onehot(givenTestDataTemp[,c("Match_type_ID"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("Match_type_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

# one-hot : Device_ID
givenTestDataTemp = givenTestData
givenTestDataTemp$Device_ID = factor(givenTestData$Device_ID)
encoder = onehot(givenTestDataTemp[,c("Device_ID"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("Device_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

# one-hot : Account_ID
givenTestDataTemp = givenTestData
givenTestDataTemp$Account_ID = factor(givenTestData$Account_ID)
encoder = onehot(givenTestDataTemp[,c("Account_ID"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("Account_ID"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

write.csv(givenTestData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/givenTestData.csv", row.names = FALSE)

# campaignIndex
twoStepDataFinal %>% group_by(Campaign_ID,campaignIndex) %>%
  summarise(count = n()) %>% View()

campaignIndexPred = twoStepDataFinal %>% group_by(Campaign_ID,campaignIndex) %>%
  summarise(count = n()) %>% select(Campaign_ID,campaignIndex) %>% as.data.frame()

givenTestData %>% left_join(campaignIndexPred, by = c("Campaign_ID")) %>%
  replace(is.na(.),0) %>% View()

givenTestData = givenTestData %>% left_join(campaignIndexPred, by = c("Campaign_ID")) %>%
  replace(is.na(.),0) %>% as.data.frame()

givenTestDataTemp = givenTestData
givenTestDataTemp$campaignIndex = factor(givenTestData$campaignIndex)
encoder = onehot(givenTestDataTemp[,c("campaignIndex"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("campaignIndex"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

# one-hot : Date
givenTestData$dayFromDate = weekdays(as.Date(givenTestData$Date))
givenTestDataTemp = givenTestData
givenTestDataTemp$dayFromDate = factor(givenTestData$dayFromDate)
encoder = onehot(givenTestDataTemp[,c("dayFromDate"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("dayFromDate"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

givenTestData %>% group_by(Date) %>% summarise(count = n()) %>% View()
twoStepDataFinal %>% group_by(Date) %>% summarise(count = n()) %>% View()

write.csv(givenTestData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/givenTestData.csv", row.names = FALSE)

# change the train data as well
# Day of the week
# one-hot : Date
twoStepDataFinal$dayFromDate = weekdays(as.Date(twoStepDataFinal$Date))
twoStepDataFinalTemp = twoStepDataFinal
twoStepDataFinalTemp$dayFromDate = factor(twoStepDataFinal$dayFromDate)
encoder = onehot(twoStepDataFinalTemp[,c("dayFromDate"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, twoStepDataFinalTemp[,c("dayFromDate"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

twoStepDataFinal = twoStepDataFinal %>% cbind(dfOneHot) %>% as.data.frame()
nrow(twoStepDataFinal)
head(twoStepDataFinal)

twoStepDataFinal = twoStepDataFinal %>% replace(is.na(.),0) %>% as.data.frame()
write.csv(twoStepDataFinal, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataFinal.csv", row.names=FALSE)

# keyADIndexX
twoStepDataFinal = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataFinal.csv")
keyADIndexXPred = twoStepDataFinal %>% group_by(Keyword_ID, Ad_group_ID, keyADIndexX) %>%
  summarise(count = n()) %>% select(-count) %>% as.data.frame()

givenTestData %>% left_join(keyADIndexXPred, by = c("Keyword_ID","Ad_group_ID")) %>% 
  replace(is.na(.),0) %>% View()

givenTestData = givenTestData %>% left_join(keyADIndexXPred, by = c("Keyword_ID","Ad_group_ID")) %>% 
  replace(is.na(.),0) %>% as.data.frame()

# keyADConversionIndex
keyADConversionIndexPred = twoStepDataFinal %>% group_by(Keyword_ID, Ad_group_ID, keyADConversionIndex) %>%
  summarise(count = n()) %>% select(-count) %>% as.data.frame()
View(keyADConversionIndexPred)

givenTestData %>% left_join(keyADConversionIndexPred, by = c("Keyword_ID","Ad_group_ID")) %>% 
  replace(is.na(.),0) %>% View()

givenTestData = givenTestData %>% left_join(keyADConversionIndexPred, by = c("Keyword_ID","Ad_group_ID")) %>% 
  replace(is.na(.),0) %>% as.data.frame()

givenTestDataTemp = givenTestData
givenTestDataTemp$keyADConversionIndex = factor(givenTestData$keyADConversionIndex)
encoder = onehot(givenTestDataTemp[,c("keyADConversionIndex"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, givenTestDataTemp[,c("keyADConversionIndex"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

givenTestData = givenTestData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(givenTestData)
head(givenTestData)

# dateIndexRevenue
dateIndexRevenuePred = twoStepDataFinal %>% group_by(Date, dateIndexRevenue) %>%
  summarise(count = n()) %>% select(Date, dateIndexRevenue) %>% as.data.frame()
View(dateIndexRevenuePred)

givenTestData %>% left_join(dateIndexRevenuePred, by = c("Date")) %>% View()

dateIndexRevenuePred$Date = as.character(dateIndexRevenuePred$Date)
class(givenTestData$Date)
class(dateIndexRevenuePred$Date)

write.csv(givenTestData, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/givenTestData.csv", row.names = FALSE)
givenTestData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/givenTestData.csv")

colTwoStepData = colnames(twoStepDataFinal)
colGivenTestData = colnames(givenTestData)
diffColumns = setdiff(colGivenTestData, colTwoStepData)
commonColumns = setdiff(colGivenTestData, diffColumns)
length(commonColumns)

givenTestData %>% group_by(indexBinary) %>% summarise(count = n()) %>% View()
twoStepDataFinal %>% group_by(indexBinary) %>% summarise(count = n()) %>% View()




####################### Step 2 : Final Model for regressing on the ##################
############################ input from the classification     ##################  

x_predictionOut = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/x_predictionOut.csv")
head(x_predictionOut)

x_predictionOut %>% group_by(Keyword_ID, Ad_group_ID) %>% summarise(count = n()) %>%
  View()

givenTestData %>% group_by(Keyword_ID, Ad_group_ID) %>% summarise(count = n()) %>%
  View()

x_predictionOut %>% filter(predictionIndex==1) %>% View()

xPredRegression = x_predictionOut %>% filter(predictionIndex==1) %>% as.data.frame()

xPredRegression %>% filter(predictionIndex==1) %>% as.data.frame() %>% View()
# 12157 positive clicks for regression

## Data for training regression
twoStepDataFinal %>% filter(rpc>0) %>% as.data.frame() %>% View() # 157401
xTrainRegression = twoStepDataFinal %>% filter(rpc>0) %>% as.data.frame()

## Model with XGBoost Regression 

xTrainRegression$logRPC = log(xTrainRegression$rpc)
hist(log(xTrainRegression$rpc)) # this is better
hist(log(xTrainRegression$logRPC))

# select columns to consider
considerColumns = c("accountIndicator", "Match_type_ID.95725474456"
                    , "Match_type_ID.872544605608", "Match_type_ID.894413617560"
                    , "Device_ID.298643508640", "Device_ID.848779586902", "Device_ID.1077718730738"
                    , "Account_ID.151664859558", "Account_ID.164144662657", "Account_ID.212779990172"
                    , "Account_ID.221354172146", "Account_ID.256188843610", "Account_ID.341124366337"
                    , "Account_ID.412971074791", "Account_ID.573604300663", "Account_ID.575525143937"
                    , "Account_ID.602182847798", "Account_ID.604905316813", "Account_ID.654870334100"
                    , "Account_ID.654870334100", "Account_ID.719583196582", "Account_ID.861287123742"
                    , "Account_ID.866124423689", "Account_ID.981453654147", "campaignIndex"
                    , "campaignIndex.0", "campaignIndex.1", "campaignIndex.4", "campaignIndex.6"
                    , "campaignIndex.8", "campaignIndex.10", "campaignIndex.20", "campaignIndex.40"
                    , "campaignIndex.60", "campaignIndex.80", "campaignIndex.100"
                    , "dayFromDate.Friday", "dayFromDate.Monday", "dayFromDate.Saturday"
                    , "dayFromDate.Sunday", "dayFromDate.Thursday", "dayFromDate.Tuesday"
                    , "dayFromDate.Wednesday", "keyADIndexX", "keyADConversionIndex"
                    , "keyADConversionIndex.1"
                    , "keyADConversionIndex.2", "keyADConversionIndex.3"
                    , "keyADConversionIndex.4", "keyADConversionIndex.5"
                    , "keyADConversionIndex.6", "keyADConversionIndex.7"
                    , "keyADConversionIndex.8", "keyADConversionIndex.9"
                    , "keyADConversionIndex.10", "keyADConversionIndex.15"
                    , "keyADConversionIndex.25", "keyADConversionIndex.50"
                    , "keyADConversionIndex.100", "keyADConversionIndex.125"
                    , "keyADConversionIndex.250") # dummy variable trap neglected

# Train vs Prediction
trainXGB = xTrainRegression[, considerColumns]
trainXGB$rpc = xTrainRegression$rpc
trainXGB$rpcLog = log(xTrainRegression$rpc)
predXGB = xPredRegression[, considerColumns]

# Train data split
intrain = caret :: createDataPartition(y=trainXGB$rpc,p=0.8,list=FALSE)
trainData = trainXGB[intrain,]
testData = trainXGB[-intrain,]

xgbTree = xgboost(data = as.matrix(select(trainData, -rpc, -rpcLog))
                  , label = trainData$rpcLog
                  , max_depth = 40
                  , eta = .05
                  , gamma = .05
                  , nthread = 5
                  , nrounds = 150
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .7
                  , colsample_bytree = .6
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

performXGB = predict(xgbTree, newdata = as.matrix(select(testData, -rpc, -rpcLog)))
# Normal
performRMSEXGB = sqrt(mean((testData$rpc - performXGB)^2))
# Log transformation
performRMSEXGB = sqrt(mean((testData$rpcLog - performXGB)^2))
print(performRMSEXGB)

# Normal
xgbAnalyse = data.frame("real" = testData$rpc
                        , "predicted" = exp(performXGB)
                        #, "Date" = testData$Date
                        )

# Log transformation
xgbAnalyse = data.frame("real" = testData$rpc
                        , "predicted" = exp(performXGB)
                        #, "Date" = testData$Date
                        )

View(xgbAnalyse)

# Train on full data set
xgbTree = xgboost(data = as.matrix(select(trainXGB, -rpc, -rpcLog))
                  , label = trainXGB$rpcLog
                  , max_depth = 40
                  , eta = .05
                  , gamma = .05
                  , nthread = 5
                  , nrounds = 150
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .7
                  , colsample_bytree = .6
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

predFinalXGB = predict(xgbTree, newdata = as.matrix(predXGB))
predXGB$predictedRPC = exp(predFinalXGB)

View(predXGB)

mainPredFrame = x_predictionOut %>% filter(predictionIndex==1) %>% as.data.frame()
View(mainPredFrame)

mainPredFrame$predictedValue = exp(predFinalXGB)
View(mainPredFrame)
mainPredFrameIndexed = mainPredFrame %>% select(index, predictedValue) %>% as.data.frame()

# Merge with main data frame
x_predictionOut %>% left_join(mainPredFrameIndexed, by = c("index")) %>%
  View()

x_predictionOutPredValue = x_predictionOut %>%
  left_join(mainPredFrameIndexed, by = c("index")) %>% replace(is.na(.),0) %>%
  as.data.frame()

indexPredictedVlaue = x_predictionOutPredValue %>% select(index, predictedValue) %>%
  as.data.frame()

givenTestData %>% left_join(indexPredictedVlaue, by = c("index")) %>% 
  replace(is.na(.),0) %>% View()


submissionData = givenTestData %>% left_join(indexPredictedVlaue, by = c("index")) %>% 
  replace(is.na(.),0) %>% select(predictedValue) %>% as.data.frame()

submissionData1 = submissionData$predictedValue

write.table( submissionData1, file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/rahman_prediction.csv"
             , sep=",",  col.names=FALSE, row.names = FALSE)

testRead = read.table(file = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/rahman_prediction.csv"
                    , header = FALSE)
View(testRead)

testRead %>% mutate(ind = case_when(.$V1 >0 ~1, .$V1 == 0 ~0)) %>%
  group_by(ind) %>% summarise(count = n()) %>% View()


