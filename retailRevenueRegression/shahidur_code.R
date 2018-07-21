library(dplyr)
library(ggplot2)
library(parallel)
library(sqldf)
library(caret)
library(onehot)
library(xgboost)
options(mc.cores = detectCores())




############################ Load the data ########################
editedData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/products_sample.csv", na.strings = c("", NA, "na", "NA", " "))
str(editedData)
head(editedData)
colnames(editedData)
head(select(editedData,-description))




############################ Exploratory analysis ############################
nrow(editedData) # 100k

## product_id
editedData %>% select(product_id) %>% distinct() %>% View() # 100k # unique key

## product_name
editedData %>% select(product_name) %>% distinct() %>% View() # ~ 42k

## brand
editedData %>% select(brand) %>% distinct() %>% View() # 784

editedData %>% group_by(brand) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View()

## category
editedData %>% select(category) %>% distinct() %>% View() # 14

editedData %>% group_by(category) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View() # accessories is the highest selling category

## currency
editedData %>% select(currency) %>% distinct() %>% View() # 2

editedData %>% group_by(currency) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View() # GBP and USD divided equally

## date
editedData %>% select(date) %>% mutate(newDate = as.Date(date)) %>% 
  select(newDate) %>% distinct() %>% arrange(desc(newDate)) %>% 
  View() # 2016 start - 2017 end

## date_first_sellout & date_first_sellthrough; NA means they never got sold out or started getting sold
editedData %>% mutate(dayDiff = difftime(as.Date(date_first_sellout),as.Date(date_first_sellthrough), units = "days")) %>%
  replace(is.na(.),0) %>% select(category, dayDiff) %>% group_by(category) %>% summarise(avgDiff = mean(dayDiff)) %>% 
  View() # give avg time taken to sell out full stock, demand indicator 

## max_discount
unique(editedData$max_discount)

editedData %>% group_by(max_discount) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View() # 0, 50, 60, 40, 30, 70% discount

## in_stock
unique(editedData$in_stock)
class(editedData$in_stock) # all are out of stock

## market
unique(editedData$market) # UK and US
editedData %>% group_by(market) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View() # US and UK market

editedData %>% group_by(market, currency) %>% summarise(count = n()) %>% arrange(desc(count)) %>%
  View() # market and currency are aligned 

## price
# check for any outliers
revenueData = editedData %>% mutate(newDate = as.Date(date)) %>% group_by(newDate, market) %>%
  summarise(totalSell = sum(price)) %>% as.data.frame()

ggplot(data = revenueData, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(~market)

ggplot(data = revenueData, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(market~., scale = "free_y") # some points are shooting above 2e+05
# maybe due to costly brand or category sales 
# 2017-07 the total sales for both the markets went down

# category wise plotting to figure out the outlier
categorySellUK = editedData %>% filter(market == "UK") %>% 
  mutate(newDate = as.Date(date)) %>% group_by(newDate, category) %>%
  summarise(totalSell = sum(price)) %>% as.data.frame()

data1 = categorySellUK %>% filter(category == "suits-sets")

ggplot(data = data1, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(category~., scale = "free_y")

## UK only
# all-in-ones is fine
# tops is fine 
# swim has an outlier between 2017-01 and 2017-02
# footwear is fine 
# dresses is fine
# accessories has outlier, multiple
# bottoms are fine 
# outerwear is fine 
# nightwear is fine 
# non-apparel has a big outlier at the end
# underwear is fine 
# hosiery is fine 
# beauty is fine 
# suits-sets is fine 

categorySellUS = editedData %>% filter(market == "US") %>% 
  mutate(newDate = as.Date(date)) %>% group_by(newDate, category) %>%
  summarise(totalSell = sum(price)) %>% as.data.frame()

data1 = categorySellUS %>% filter(category == "suits-sets")

ggplot(data = data1, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(category~., scale = "free_y")

## US only
# all-in-ones is fine
# tops is fine 
# swim has an outlier between 2017-01 and 2017-02; same as UK
# footwear is fine 
# dresses is fine
# accessories has outlier, multiple
# bottoms are fine 
# outerwear is fine 
# nightwear is fine 
# non-apparel has a big outlier at the end
# underwear has an outlier
# hosiery is fine 
# beauty is fine 
# suits-sets is fine 

## shop
unique(editedData$shop) # 1




######################## Feature Engineering ########################

#### New features
## Newdate
editedData = editedData %>% mutate(newDate = as.Date(date)) %>% as.data.frame()

## dayDiff
editedData = editedData %>% mutate(dayDiff = difftime(as.Date(date_first_sellout),as.Date(date_first_sellthrough), units = "days")) %>% 
  as.data.frame()

## -description
editedData = editedData %>% select(-description) %>% as.data.frame()

## discountedPrice
editedData %>% mutate(discountedPrice = round((1-max_discount)* price,2)) %>% View()
editedData = editedData %>% mutate(discountedPrice = round((1-max_discount)* price,2)) %>% 
  as.data.frame()

#write.csv(editedData, file ="/Users/sheikhshahidurrahman/Documents/DS/editedDataNew.csv")

#### Detect outliers in UK Data
categorySellUK = editedData %>% filter(market == "UK") %>%
  mutate(newDate = as.Date(date)) %>% group_by(newDate, category) %>%
  summarise(totalSell = sum(price)) %>% as.data.frame()

data1 = categorySellUK %>% filter(category == "non-apparel")

ggplot(data = data1, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(category~., scale = "free_y")

data2 = categorySellUK %>% filter(category == "non-apparel") %>% 
  filter(totalSell>1e+05)
print(data2)

# 2017-01-20     swim  14949.62 
#      newDate    category totalSell
# 1 2017-05-03 accessories    308315
# 2 2017-06-13 accessories    277065
# 3 2017-11-30 accessories    334145
# 4 2017-12-12 accessories    209800
# 2017-11-08 non-apparel    142000

## Detect outliers in US Data
categorySellUS = editedData %>% filter(market == "US") %>% 
  mutate(newDate = as.Date(date)) %>% group_by(newDate, category) %>%
  summarise(totalSell = sum(price)) %>% as.data.frame()

data1 = categorySellUS %>% filter(category == "underwear")

ggplot(data = data1, aes(x = newDate, y = totalSell)) +
  geom_line() + facet_grid(category~., scale = "free_y")

data2 = categorySellUS %>% filter(category == "underwear") %>% 
  filter(totalSell>2000)
print(data2)

# 2017-01-20     swim   17071.4
#      newDate    category totalSell
# 1 2017-10-12 accessories    295900
# 2 2017-11-30 accessories    279210
# 3 2017-12-12 accessories    267105
# 2017-11-09 non-apparel    185433
# 2017-04-29 underwear    3996.5

#### Remove the outliers; for simplicity, removing the row instead of amputation
editedDataNew = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/editedDataNew.csv", na.strings = c("", NA, "na", "NA", " "))
editedDataNew = editedDataNew %>% select(-X)
nrow(editedDataNew)

# swim
editedDataOR = editedDataNew %>% filter(market == 'UK') %>% as.data.frame()
xx = sqldf("select * from editedDataOR where market = 'UK' and newDate = '2017-01-20'
       and category = 'swim' ")
editedDataOR = editedDataOR %>% anti_join(xx, by = c("product_id"))

# accessories 
xx = sqldf("select * from editedDataOR where market = 'UK' and 
            newDate in ('2017-05-03', '2017-06-13', '2017-11-30', '2017-12-12')
            and category = 'accessories' ")
editedDataOR = editedDataOR %>% anti_join(xx, by = c("product_id"))

# non-apparel
xx = sqldf("select * from editedDataOR where market = 'UK' and newDate = '2017-11-08'
       and category = 'non-apparel' ")
editedDataOR = editedDataOR %>% anti_join(xx, by = c("product_id"))

write.csv(editedDataOR, file = "/Users/sheikhshahidurrahman/Documents/DS/editedDataOR.csv", row.names = FALSE)




############################ Modelling ############################

#### For our use case we're going forward with the UK data only

#### Use case 1 : Finding the date for date_first_sellout for brands having only #### 
# date_first_sellthrough

editedDataOR %>% group_by(max_discount) %>% summarise(count = n()) %>% 
  as.data.frame()%>% View()

editedDataOR = editedDataOR %>% filter(max_discount %in% c(0.0000, 0.3000, 0.4000, 0.5000
                                            , 0.6000, 0.7000, 0.8000)) %>%
  as.data.frame()
# 50043 - 147
# 82+65 = 147
# 49896

## Considerable data
editedDataOR %>% group_by(dayDiff) %>% summarise(count = n()) %>% View()
considerData = editedDataOR %>% filter(!is.na(dayDiff)) # 27651

write.csv(considerData, file = "/Users/sheikhshahidurrahman/Documents/DS/considerData.csv"
          , row.names = FALSE)

modelData = considerData %>% select(category, max_discount, price, dayDiff)

## Convert category to one-hot vectors 
tempData = modelData
tempData$category = factor(modelData$category)
encoder = onehot(tempData[,c("category"), drop = FALSE], max_levels = 1000)
oneHotPredict = predict(encoder, tempData[,c("category"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

modelData = modelData %>% cbind(dfOneHot) %>% as.data.frame()
nrow(modelData)
View(modelData)

write.csv(modelData, file = "/Users/sheikhshahidurrahman/Documents/DS/modelData.csv"
          , row.names = FALSE)
modelData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/modelData.csv")
colnames(modelData)
modelData = modelData %>% select(-category)
modelData$logDayDiff = log(modelData$dayDiff+.01)

## Test Train split
set.seed(123)

intrain = caret :: createDataPartition(y=modelData$dayDiff, p=0.8, list=FALSE)
trainData = modelData[intrain,]
testData = modelData[-intrain,]
str(trainData)
str(testData)

xgbTree = xgboost(data = as.matrix(select(trainData, -dayDiff, -category.suits.sets, -logDayDiff, -discountedPrice))
                  , label = trainData$dayDiff
                  , max_depth = 8
                  , eta = .1
                  , gamma = 3
                  , nthread = 5
                  , nrounds = 150
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .7
                  , colsample_bytree = .6
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

performXGB = predict(xgbTree, newdata = as.matrix(select(testData, -dayDiff, -category.suits.sets, -logDayDiff, -discountedPrice)))

# Normal
performRMSEXGB = sqrt(mean((testData$dayDiff - (performXGB))^2))
print(performRMSEXGB)
# Log transformation
performRMSEXGBLog = sqrt(mean((testData$logDayDiff - performXGB)^2))
print(performRMSEXGBLog)

# Train RMSE 55.698807
# Test RMSE 62.72786 # the bias variance tradeoff is balanced




#### Use case 2 : Forecasting of daily revenue value by category #### 
#### In this case, I'm in applying time series forecasting; the data is not suitable for that

editedDataOR = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/editedDataOR.csv")

## Assumption : if date_first_sellthrough is null then the order is not placed
model2Data = editedDataOR %>% filter(!is.na(date_first_sellthrough)) %>%
  as.data.frame() # 34197

model2Data = model2Data %>% group_by(newDate, category) %>% 
  summarise(revenue = sum(discountedPrice)
            , dayDiffAvg = mean(dayDiff)
  ) %>% replace(is.na(.),0) %>% as.data.frame()


# One hot encoding for day of week
model2Data$dayOfWeek = weekdays(as.Date(model2Data$newDate))

tempData = model2Data
tempData$dayOfWeek = factor(model2Data$dayOfWeek)
encoder = onehot(tempData[,c("dayOfWeek"), drop = FALSE], max_levels = 1000)
oneHotPredict = predict(encoder, tempData[,c("dayOfWeek"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

model2Data = model2Data %>% cbind(dfOneHot) %>% as.data.frame()
nrow(model2Data)
View(model2Data)

tempData = model2Data
tempData$category = factor(model2Data$category)
encoder = onehot(tempData[,c("category"), drop = FALSE], max_levels = 1000)
oneHotPredict = predict(encoder, tempData[,c("category"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

model2Data = model2Data %>% cbind(dfOneHot) %>% as.data.frame()
nrow(model2Data)
View(model2Data)

write.csv(model2Data, file = "/Users/sheikhshahidurrahman/Documents/DS/model2Data.csv"
          , row.names = FALSE)
model2Data = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/model2Data.csv")
colnames(model2Data)

model2Data = model2Data %>% select(-category, -dayOfWeek)
model2Data$logRevenue = log(model2Data$revenue)

## Test Train split
set.seed(456)

intrain = caret :: createDataPartition(y=model2Data$revenue, p=0.8, list=FALSE)
trainData = model2Data[intrain,]
testData = model2Data[-intrain,]
str(trainData)
str(testData)

xgbTree = xgboost(data = as.matrix(select(trainData, -revenue, -logRevenue, -newDate))
                  , label = trainData$revenue
                  , max_depth = 40
                  , eta = .05
                  , gamma = 3
                  , nthread = 5
                  , nrounds = 500
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = 1
                  , colsample_bytree = 1
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

performXGB = predict(xgbTree, newdata = as.matrix(select(testData, -revenue, -logRevenue, -newDate)))

# Normal
performRMSEXGB = sqrt(mean((testData$revenue - (performXGB))^2))
print(performRMSEXGB)
# Log transformation
performRMSEXGBLog = sqrt(mean((testData$logRevenue - performXGB)^2))
print(performRMSEXGBLog)

# Train RMSE 2974.081
# Test RMSE 3282.199 # the bias variance tradeoff is quite balanced

model2plot = data.frame(indexreal = testData$revenue
                        , predicted = performXGB
                        , Date = testData$newDate)


## Visualization of real vs predicted
ggplot(data = model2plot,aes(as.Date(Date))) +
  geom_line(aes(y=predicted), colour = "red") + 
  geom_line(aes(y=real), colour = "blue") + 
  xlab("Date") + 
  ylab("Revenue") +
  ggtitle("Prediction=Blue vs Real=Red") +
  scale_y_continuous(limits = c(0,15000))

# mdf = reshape2::melt(model2plot, id.var = "Date")
# 
# ggplot(mdf, aes(x = Date, y = value, colour = variable)) + 
#   geom_line()





