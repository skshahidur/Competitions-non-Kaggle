library(dplyr)
library(ggplot2)
library(parallel)
library(caret)
library(onehot)
library(xgboost)
library(reshape2)
options(mc.cores = detectCores())


############################ Load data and Exploratory analysis ########################
ssData = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/Lead_testdata.csv"
                  , sep = ";"
                  , na.strings = c("", " ", "NA", "na",NA)
                  , dec = ",")
View(ssData)
nrow(ssData) # 23245
ncol(ssData) # 53

## Target_Sold
unique(ssData$Target_Sold)
ssData %>% group_by(Target_Sold) %>% summarise(count = n()) %>% View()
# Target_Sold count
# 0       20117
# 1       3128

## Target_Sales
unique(ssData$Target_Sales)
class(ssData$Target_Sales)
ssData %>% group_by(Target_Sales) %>% summarise(count = n()) %>% View() # 20117 NA
boxplot(as.numeric(ssData$Target_Sales))
hist(as.numeric(ssData$Target_Sales)) # outlier in model perspective, three 197848 or log transform
boxplot(log(ssData$Target_Sales))
ssData %>% filter(Target_Sales>150000) %>% View()
median(ssData$Target_Sales, na.rm = TRUE) # 1662.4
mean(ssData$Target_Sales, na.rm = TRUE) # 6029.586

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

Mode(ssData$Target_Sales, na.rm = TRUE) # 1195

# outlier should be capped to 156828 value 

## Var_04
unique(ssData$Var_04)
class(ssData$Var_04)
ssData %>% group_by(Var_04) %>% summarise(count = n()) %>% View()
ssData %>% group_by(Var_04, Target_Sold) %>% summarise(count = n()) %>% View()
# Var_04==1 has highest rate of conversion

## No_Act_1M
ssData %>% group_by(No_Act_1M) %>% summarise(count = n()) %>% View() # 13803 NA
boxplot(ssData$No_Act_1M)
ssData %>% group_by(No_Act_1M, Target_Sold) %>% summarise(count = n()) %>% View() # 1202 from before 1 month

## No_Act_3M
boxplot(ssData$No_Act_3M)
ssData %>% group_by(No_Act_3M) %>% summarise(count = n()) %>% View()
# take 3M - 1M for 2M activity and these will be mutually exclusive to each other

## No_Act_6M
boxplot(ssData$No_Act_6M)
ssData %>% group_by(No_Act_6M) %>% summarise(count = n()) %>% View()
# take 6M - 3M for 3M activity and these will be mutually exclusive to each other

## No_Act_Lo_1M
unique(ssData$No_Act_Lo_1M)
boxplot(ssData$No_Act_Lo_1M)
ssData %>% group_by(No_Act_Lo_1M) %>% summarise(count = n()) %>% View()

## No_Act_Lo_1vs2
unique(ssData$No_Act_Lo_1vs2)
boxplot(ssData$No_Act_Lo_1vs2)
ssData %>% group_by(No_Act_Lo_1vs2) %>% summarise(count = n()) %>% View() # 18528 NA

## No_Act_Ph_1M
unique(ssData$No_Act_Ph_1M)
boxplot(ssData$No_Act_Ph_1M)
ssData %>% group_by(No_Act_Ph_1M) %>% summarise(count = n()) %>% View() # 13803 NA
# minus the call+email to get only email

## No_Act_Ph_3M
unique(ssData$No_Act_Ph_3M)
boxplot(ssData$No_Act_Ph_3M)
ssData %>% group_by(No_Act_Ph_3M) %>% summarise(count = n()) %>% View() # 6960 NA

## No_Act_WOF_1M
unique(ssData$No_Act_WOF_1M)
boxplot(ssData$No_Act_WOF_1M)
ssData %>% group_by(No_Act_WOF_1M) %>% summarise(count = n()) %>% View() # 13803 NA
# No_Act_1M != No_Act_Lo_1M + No_Act_WOF_1M

## No_Act_WOF_1vs2
unique(ssData$No_Act_WOF_1vs2)
class(ssData$No_Act_WOF_1vs2)
boxplot(ssData$No_Act_WOF_1vs2)
ssData %>% group_by(No_Act_WOF_1vs2) %>% summarise(count = n()) %>% View() # 18528 NA

## No_Act_WOS_12M
unique(ssData$No_Act_WOS_12M)
class(ssData$No_Act_WOS_12M)
boxplot(ssData$No_Act_WOS_12M)
ssData %>% group_by(No_Act_WOS_12M) %>% summarise(count = n()) %>% View() # 2943 NA

## No_list_FB_6M
unique(ssData$No_list_FB_6M)
class(ssData$No_list_FB_6M)
boxplot(ssData$No_list_FB_6M)
ssData %>% group_by(No_list_FB_6M) %>% summarise(count = n()) %>% View() # 0 NA

## No_list_IND_24M
unique(ssData$No_list_IND_24M)
class(ssData$No_list_IND_24M)
boxplot(ssData$No_list_IND_24M)
hist(ssData$No_list_IND_24M)
ssData %>% group_by(No_list_IND_24M) %>% summarise(count = n()) %>% View() # 3657 NA

## No_List_Off_1M
unique(ssData$No_List_Off_1M)
class(ssData$No_List_Off_1M)
boxplot(ssData$No_List_Off_1M)
hist(ssData$No_List_Off_1M)
ssData %>% group_by(No_List_Off_1M) %>% summarise(count = n()) %>% View() # 21937 NA

## No_List_On_24M # only the last month?
unique(ssData$No_List_On_24M)
class(ssData$No_List_On_24M)
boxplot(ssData$No_List_On_24M)
hist(ssData$No_List_On_24M)
ssData %>% group_by(No_List_On_24M) %>% summarise(count = n()) %>% View() # 5817 NA

## No_List_STST_3M
unique(ssData$No_List_STST_3M)
class(ssData$No_List_STST_3M)
boxplot(ssData$No_List_STST_3M)
hist(ssData$No_List_STST_3M)
ssData %>% group_by(No_List_STST_3M) %>% summarise(count = n()) %>% View() # 15356 NA

## No_List_STST_6M
unique(ssData$No_List_STST_6M)
class(ssData$No_List_STST_6M)
boxplot(ssData$No_List_STST_6M)
hist(ssData$No_List_STST_6M)
ssData %>% group_by(No_List_STST_6M) %>% summarise(count = n()) %>% View() # 14252 NA

## No_List_T7_1M
unique(ssData$No_List_T7_1M)
class(ssData$No_List_T7_1M)
boxplot(ssData$No_List_T7_1M)
hist(ssData$No_List_T7_1M)
ssData %>% group_by(No_List_T7_1M) %>% summarise(count = n()) %>% View() # 17091 NA

## Avg_OAF_6M
unique(ssData$Avg_OAF_6M)
class(ssData$Avg_OAF_6M)
boxplot(ssData$Avg_OAF_6M)
hist(ssData$Avg_OAF_6M)
ssData %>% group_by(Avg_OAF_6M) %>% summarise(count = n()) %>% View() # 13585 NA

## AVG_Refresh_p_Posting
unique(ssData$AVG_Refresh_p_Posting)
class(ssData$AVG_Refresh_p_Posting)
boxplot(ssData$AVG_Refresh_p_Posting)
hist(ssData$AVG_Refresh_p_Posting)
ssData %>% group_by(AVG_Refresh_p_Posting) %>% summarise(count = n()) %>% View() # 8920 NA

## AVG_Renew_p_Posting
unique(ssData$AVG_Renew_p_Posting)
class(ssData$AVG_Renew_p_Posting)
boxplot(ssData$AVG_Renew_p_Posting)
hist(ssData$AVG_Renew_p_Posting)
ssData %>% group_by(AVG_Renew_p_Posting) %>% summarise(count = n()) %>% View() # 8920 NA

## AVG_Share_Online
unique(ssData$AVG_Share_Online)
class(ssData$AVG_Share_Online)
boxplot(ssData$AVG_Share_Online)
hist(ssData$AVG_Share_Online)
ssData %>% group_by(AVG_Share_Online) %>% summarise(count = n()) %>% View() # 8920 NA

## AVG_STF_12M
unique(ssData$AVG_STF_12M)
class(ssData$AVG_STF_12M)
boxplot(ssData$AVG_STF_12M)
hist(ssData$AVG_STF_12M)
ssData %>% group_by(AVG_STF_12M) %>% summarise(count = n()) %>% View() # 12275 NA

## AVG_STF_1M
unique(ssData$AVG_STF_1M)
class(ssData$AVG_STF_1M)
boxplot(ssData$AVG_STF_1M)
hist(ssData$AVG_STF_1M)
ssData %>% group_by(AVG_STF_1M) %>% summarise(count = n()) %>% View() # 15587 NA

## AVG_STF_6M
unique(ssData$AVG_STF_6M)
class(ssData$AVG_STF_6M)
boxplot(ssData$AVG_STF_6M)
hist(ssData$AVG_STF_6M)
ssData %>% group_by(AVG_STF_6M) %>% summarise(count = n()) %>% View() # 13585 NA

## Avg_Views_12M
unique(ssData$Avg_Views_12M)
class(ssData$Avg_Views_12M)
boxplot(ssData$Avg_Views_12M)
hist(ssData$Avg_Views_12M)
ssData %>% group_by(Avg_Views_12M) %>% summarise(count = n()) %>% View() # 12275 NA

## Avg_Views_1M
unique(ssData$Avg_Views_1M)
class(ssData$Avg_Views_1M)
boxplot(ssData$Avg_Views_1M)
hist(ssData$Avg_Views_1M)
ssData %>% group_by(Avg_Views_1M) %>% summarise(count = n()) %>% View() # 15587 NA

## Avg_Views_6M
unique(ssData$Avg_Views_6M)
class(ssData$Avg_Views_6M)
boxplot(ssData$Avg_Views_6M)
hist(ssData$Avg_Views_6M)
ssData %>% group_by(Avg_Views_6M) %>% summarise(count = n()) %>% View() # 13585 NA

## Count_SIC_3
unique(ssData$Count_SIC_3)
class(ssData$Count_SIC_3)
boxplot(ssData$Count_SIC_3)
hist(ssData$Count_SIC_3)
ssData %>% group_by(Count_SIC_3) %>% summarise(count = n()) %>% View() # 1987 NA

## Cre_Val_12M
unique(ssData$Cre_Val_12M)
class(ssData$Cre_Val_12M)
boxplot(ssData$Cre_Val_12M)
hist(ssData$Cre_Val_12M)
ssData %>% group_by(Cre_Val_12M) %>% summarise(count = n()) %>% View() # 14448 NA

## Cre_Val_1M
unique(ssData$Cre_Val_1M)
class(ssData$Cre_Val_1M)
boxplot(ssData$Cre_Val_1M)
hist(ssData$Cre_Val_1M)
ssData %>% group_by(Cre_Val_1M) %>% summarise(count = n()) %>% View() # 18065 NA

## Cre_Val_6M
unique(ssData$Cre_Val_6M)
class(ssData$Cre_Val_6M)
boxplot(ssData$Cre_Val_6M)
hist(ssData$Cre_Val_6M)
ssData %>% group_by(Cre_Val_6M) %>% summarise(count = n()) %>% View() # 15718 NA

## Inv_Days_12M
unique(ssData$Inv_Days_12M)
class(ssData$Inv_Days_12M)
boxplot(ssData$Inv_Days_12M)
hist(ssData$Inv_Days_12M)
ssData %>% group_by(Inv_Days_12M) %>% summarise(count = n()) %>% View() # 15332 NA

## Inv_Days_1M
unique(ssData$Inv_Days_1M)
class(ssData$Inv_Days_1M)
boxplot(ssData$Inv_Days_1M)
hist(ssData$Inv_Days_1M)
ssData %>% group_by(Inv_Days_1M) %>% summarise(count = n()) %>% View() # 20568 NA

## Inv_Days_6M
unique(ssData$Inv_Days_6M)
class(ssData$Inv_Days_6M)
boxplot(ssData$Inv_Days_6M)
hist(ssData$Inv_Days_6M)
ssData %>% group_by(Inv_Days_6M) %>% summarise(count = n()) %>% View() # 17153 NA

## Inv_Val_12M
unique(ssData$Inv_Val_12M)
class(ssData$Inv_Val_12M)
boxplot(ssData$Inv_Val_12M)
hist(ssData$Inv_Val_12M)
ssData %>% group_by(Inv_Val_12M) %>% summarise(count = n()) %>% View() # 15332 NA

## Inv_Val_1M
unique(ssData$Inv_Val_1M)
class(ssData$Inv_Val_1M)
boxplot(ssData$Inv_Val_1M)
hist(ssData$Inv_Val_1M)
ssData %>% group_by(Inv_Val_1M) %>% summarise(count = n()) %>% View() # 20568 NA

## Inv_Val_3M
unique(ssData$Inv_Val_3M)
class(ssData$Inv_Val_3M)
boxplot(ssData$Inv_Val_3M)
hist(ssData$Inv_Val_3M)
ssData %>% group_by(Inv_Val_3M) %>% summarise(count = n()) %>% View() # 18911 NA

## List_FB_HOM_1M
unique(ssData$List_FB_HOM_1M)
class(ssData$List_FB_HOM_1M)
boxplot(ssData$List_FB_HOM_1M)
hist(ssData$List_FB_HOM_1M)
ssData %>% group_by(List_FB_HOM_1M) %>% summarise(count = n()) %>% View() # 15640 NA

## List_FB_off_6M
unique(ssData$List_FB_off_6M)
class(ssData$List_FB_off_6M)
boxplot(ssData$List_FB_off_6M)
hist(ssData$List_FB_off_6M)
ssData %>% group_by(List_FB_off_6M) %>% summarise(count = n()) %>% View() # 19586 NA

## List_STS_FB_1M
unique(ssData$List_STS_FB_1M)
class(ssData$List_STS_FB_1M)
boxplot(ssData$List_STS_FB_1M)
hist(ssData$List_STS_FB_1M)
ssData %>% group_by(List_STS_FB_1M) %>% summarise(count = n()) %>% View() # 16488 NA

## List_STS_Index_12M
unique(ssData$List_STS_Index_12M)
class(ssData$List_STS_Index_12M)
boxplot(ssData$List_STS_Index_12M)
hist(ssData$List_STS_Index_12M)
ssData %>% group_by(List_STS_Index_12M) %>% summarise(count = n()) %>% View() # 12918 NA

## List_STS_Index_1M
unique(ssData$List_STS_Index_1M)
class(ssData$List_STS_Index_1M)
boxplot(ssData$List_STS_Index_1M)
hist(ssData$List_STS_Index_1M)
ssData %>% group_by(List_STS_Index_1M) %>% summarise(count = n()) %>% View() # 16497 NA

## List_STS_Index_3M
unique(ssData$List_STS_Index_3M)
class(ssData$List_STS_Index_3M)
boxplot(ssData$List_STS_Index_3M)
hist(ssData$List_STS_Index_3M)
ssData %>% group_by(List_STS_Index_3M) %>% summarise(count = n()) %>% View() # 15364 NA

## List_STS_off_24M
unique(ssData$List_STS_off_24M)
class(ssData$List_STS_off_24M)
boxplot(ssData$List_STS_off_24M)
hist(ssData$List_STS_off_24M)
ssData %>% group_by(List_STS_off_24M) %>% summarise(count = n()) %>% View() # 18681 NA

## List_STS_T4_12M
unique(ssData$List_STS_T4_12M)
class(ssData$List_STS_T4_12M)
boxplot(ssData$List_STS_T4_12M)
hist(ssData$List_STS_T4_12M)
ssData %>% group_by(List_STS_T4_12M) %>% summarise(count = n()) %>% View() # 17039 NA

## MISC_Days_Since_Last_act_noffer
unique(ssData$MISC_Days_Since_Last_act_noffer)
class(ssData$MISC_Days_Since_Last_act_noffer)
boxplot(ssData$MISC_Days_Since_Last_act_noffer)
hist(ssData$MISC_Days_Since_Last_act_noffer)
ssData %>% group_by(MISC_Days_Since_Last_act_noffer) %>% summarise(count = n()) %>% View() # 1842 NA

## MISC_Days_Since_Last_Offer
unique(ssData$MISC_Days_Since_Last_Offer)
class(ssData$MISC_Days_Since_Last_Offer)
boxplot(ssData$MISC_Days_Since_Last_Offer)
hist(ssData$MISC_Days_Since_Last_Offer)
ssData %>% group_by(MISC_Days_Since_Last_Offer) %>% summarise(count = n()) %>% View() # 9295 NA

## Offer_days_1M
unique(ssData$Offer_days_1M)
class(ssData$Offer_days_1M)
boxplot(ssData$Offer_days_1M)
hist(ssData$Offer_days_1M)
ssData %>% group_by(Offer_days_1M) %>% summarise(count = n()) %>% View() # 19277 NA

## Offer_days_24M
unique(ssData$Offer_days_24M)
class(ssData$Offer_days_24M)
boxplot(ssData$Offer_days_24M)
hist(ssData$Offer_days_24M)
ssData %>% group_by(Offer_days_24M) %>% summarise(count = n()) %>% View() # 9961 NA




############################ Feature Engineering ########################




#### New feature Generation ####

ssDataOR = ssData

## logTargetSales
ssDataOR$logTargetSales = log(ssData$Target_Sales)

## No_Act_1M
ssDataOR %>% mutate(No_Act_1M_capped = ifelse(No_Act_1M>60,60,No_Act_1M)) %>%
  select(No_Act_1M, No_Act_1M_capped) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_1M_capped = ifelse(No_Act_1M>60,60,No_Act_1M)) %>%
  as.data.frame()

## No_Act_1M_no_Ph
ssDataOR$No_Act_1M_no_Ph = ssDataOR$No_Act_1M - ssDataOR$No_Act_Ph_1M

ssDataOR %>% select(No_Act_1M, No_Act_Ph_1M, No_Act_1M_no_Ph) %>% View()
ssDataOR %>% group_by(No_Act_1M, No_Act_Ph_1M) %>% summarise(count = n()) %>% View()

## No_Act_3M_no_Ph
# No_Act_3M - No_Act_Ph_3M - (No_Act_1M - No_Act_Ph_1M)
ssDataOR$No_Act_2M_no_Ph = ssDataOR$No_Act_3M - 
  ssDataOR$No_Act_Ph_3M - 
  ifelse(is.na(ssDataOR$No_Act_1M),0,ssDataOR$No_Act_1M) + 
  ifelse(is.na(ssDataOR$No_Act_Ph_1M),0,ssDataOR$No_Act_Ph_1M)

ssDataOR %>% select(No_Act_3M, No_Act_Ph_3M, No_Act_1M, No_Act_Ph_1M, No_Act_2M_no_Ph) %>% View()

## No_Act_3M_all
ssDataOR$No_Act_3M_all = ssDataOR$No_Act_6M - 
  ifelse(is.na(ssDataOR$No_Act_3M),0,ssDataOR$No_Act_3M)

ssDataOR %>% select(No_Act_6M, No_Act_3M, No_Act_3M_all) %>% View()

## No_Act_Lo_1vs2_2nd
ssDataOR$No_Act_Lo_1vs2_2nd = ssDataOR$No_Act_Lo_1M - 
  ifelse(is.na(ssDataOR$No_Act_Lo_1vs2),0,ssDataOR$No_Act_Lo_1vs2)

ssDataOR %>% select(No_Act_Lo_1M, No_Act_Lo_1vs2, No_Act_Lo_1vs2_2nd) %>% View()

## No_Act_Ph_2M
ssDataOR$No_Act_Ph_2M = ssDataOR$No_Act_Ph_3M - 
  ifelse(is.na(ssDataOR$No_Act_Ph_1M),0,ssDataOR$No_Act_Ph_1M)

ssDataOR %>% select(No_Act_Ph_3M, No_Act_Ph_1M, No_Act_Ph_2M) %>% View()

## No_Act_WOF_1vs2_2nd
ssDataOR$No_Act_WOF_1vs2_2nd = ssDataOR$No_Act_WOF_1M - 
  ifelse(is.na(ssDataOR$No_Act_WOF_1vs2),NA,ssDataOR$No_Act_WOF_1vs2)

ssDataOR %>% select(No_Act_WOF_1M, No_Act_WOF_1vs2, No_Act_WOF_1vs2_2nd) %>% View()

## No_Act_WOF_10M
ssDataOR$No_Act_WOF_10M = ssDataOR$No_Act_WOS_12M - 
  ifelse(is.na(ssDataOR$No_Act_WOF_1vs2_2nd),0,ssDataOR$No_Act_WOF_1vs2_2nd) - 
  ifelse(is.na(ssDataOR$No_Act_WOF_1M),0,ssDataOR$No_Act_WOF_1M)

ssDataOR %>% select(No_Act_WOS_12M, No_Act_WOF_1M, No_Act_WOF_1vs2_2nd, No_Act_WOF_10M) %>% View()

## AVG_STF_6M_last # actually we can't do this as we don't know the base customer number
ssDataOR$AVG_STF_6M_last = (12*ssDataOR$AVG_STF_12M - 
  6*ifelse(is.na(ssDataOR$AVG_STF_6M),0,ssDataOR$AVG_STF_6M))/6

ssDataOR %>% select(AVG_STF_12M, AVG_STF_6M, AVG_STF_6M_last) %>% View()

## AVG_STF_5M
ssDataOR$AVG_STF_5M = (6*ssDataOR$AVG_STF_6M - 
  ifelse(is.na(ssDataOR$AVG_STF_1M),0,ssDataOR$AVG_STF_1M))/5

ssDataOR %>% select(AVG_STF_6M, AVG_STF_1M, AVG_STF_5M) %>% View()

## Avg_Views_6M_last
ssDataOR$Avg_Views_6M_last = (12*ssDataOR$Avg_Views_12M - 
                              6*ifelse(is.na(ssDataOR$Avg_Views_6M),0,ssDataOR$Avg_Views_6M))/6

ssDataOR %>% select(Avg_Views_12M, Avg_Views_6M, Avg_Views_6M_last) %>% View()

## Avg_Views_5M
ssDataOR$Avg_Views_5M = (6*ssDataOR$Avg_Views_6M - 
                         ifelse(is.na(ssDataOR$Avg_Views_1M),0,ssDataOR$Avg_Views_1M))/5

ssDataOR %>% select(Avg_Views_6M, Avg_Views_1M, Avg_Views_5M) %>% View()

## Cre_Val_6M_last
ssDataOR$Cre_Val_6M_last = ssDataOR$Cre_Val_12M - 
                                ifelse(is.na(ssDataOR$Cre_Val_6M),0,ssDataOR$Cre_Val_6M)

ssDataOR %>% select(Cre_Val_12M, Cre_Val_6M, Cre_Val_6M_last) %>% View()

## Cre_Val_5M
ssDataOR$Cre_Val_5M = ssDataOR$Cre_Val_6M - 
                           ifelse(is.na(ssDataOR$Cre_Val_1M),0,ssDataOR$Cre_Val_1M)

ssDataOR %>% select(Cre_Val_6M, Cre_Val_1M, Cre_Val_5M) %>% View()

## Inv_Days_6M_last
ssDataOR$Inv_Days_6M_last = ssDataOR$Inv_Days_12M - 
  ifelse(is.na(ssDataOR$Inv_Days_6M),0,ssDataOR$Inv_Days_6M)

ssDataOR %>% select(Inv_Days_12M, Inv_Days_6M, Inv_Days_6M_last) %>% View()

## Inv_Days_5M
ssDataOR$Inv_Days_5M = ssDataOR$Inv_Days_6M - 
  ifelse(is.na(ssDataOR$Inv_Days_1M),0,ssDataOR$Inv_Days_1M)

ssDataOR %>% select(Inv_Days_6M, Inv_Days_1M, Inv_Days_5M) %>% View()

## Inv_Val_6M_last
ssDataOR$Inv_Val_6M_last = ssDataOR$Inv_Val_12M - 
  ifelse(is.na(ssDataOR$Inv_Val_3M),0,ssDataOR$Inv_Val_3M)

ssDataOR %>% select(Inv_Val_12M, Inv_Val_3M, Inv_Val_6M_last) %>% View()

## Inv_Val_2M
ssDataOR$Inv_Val_2M = ssDataOR$Inv_Val_3M - 
  ifelse(is.na(ssDataOR$Inv_Val_1M),0,ssDataOR$Inv_Val_1M)

ssDataOR %>% select(Inv_Val_3M, Inv_Val_1M, Inv_Val_2M) %>% View()

## Offer_days_23M
ssDataOR$Offer_days_23M = ssDataOR$Offer_days_24M - 
  ifelse(is.na(ssDataOR$Offer_days_1M),0,ssDataOR$Offer_days_1M)

ssDataOR %>% select(Offer_days_24M, Offer_days_1M, Offer_days_23M) %>% View()

write.csv(ssDataOR, file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataOR.csv"
          , row.names = FALSE)
ssDataOR = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataOR.csv"
                    , na.strings = c("", " ", "NA", "na",NA))




#### Outlier Removal : Capping the outliers ####
## Target_Sales_OR
ssDataOR %>% mutate(Target_Sales_OR = ifelse(Target_Sales>=70000,70000,Target_Sales)) %>%
  select(Target_Sales_OR, Target_Sales) %>% View()

ssDataOR = ssDataOR %>% mutate(Target_Sales_OR = ifelse(Target_Sales>=70000,70000,Target_Sales)) %>%
  as.data.frame()

## No_Act_1M_no_Ph_OR
ssDataOR %>% group_by(No_Act_1M_no_Ph) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_1M_no_Ph_OR = ifelse(No_Act_1M_no_Ph>=66,66,No_Act_1M_no_Ph)) %>%
  select(No_Act_1M_no_Ph, No_Act_1M_no_Ph_OR) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_1M_no_Ph_OR = ifelse(No_Act_1M_no_Ph>=66,66,No_Act_1M_no_Ph)) %>%
  as.data.frame()

## No_Act_2M_no_Ph_OR
ssDataOR %>% group_by(No_Act_2M_no_Ph) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_2M_no_Ph_OR = ifelse(No_Act_2M_no_Ph>=100,100,No_Act_2M_no_Ph)) %>%
  select(No_Act_2M_no_Ph_OR, No_Act_2M_no_Ph) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_2M_no_Ph_OR = ifelse(No_Act_2M_no_Ph>=100,100,No_Act_2M_no_Ph)) %>%
  as.data.frame()

## No_Act_3M_all_OR
ssDataOR %>% group_by(No_Act_3M_all) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_3M_all_OR = ifelse(No_Act_3M_all>=100,100,No_Act_3M_all)) %>%
  select(No_Act_3M_all_OR, No_Act_3M_all) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_3M_all_OR = ifelse(No_Act_3M_all>=100,100,No_Act_3M_all)) %>%
  as.data.frame()

## No_Act_Lo_1M_OR
ssDataOR %>% group_by(No_Act_Lo_1M) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_Lo_1M_OR = ifelse(No_Act_Lo_1M>=35,35,No_Act_Lo_1M)) %>%
  select(No_Act_Lo_1M_OR, No_Act_Lo_1M) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_Lo_1M_OR = ifelse(No_Act_Lo_1M>=35,35,No_Act_Lo_1M)) %>%
  as.data.frame()

## No_Act_Lo_1vs2_2nd_OR
ssDataOR %>% group_by(No_Act_Lo_1vs2_2nd) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_Lo_1vs2_2nd_OR = ifelse(No_Act_Lo_1vs2_2nd>=35,35,No_Act_Lo_1vs2_2nd)) %>%
  select(No_Act_Lo_1vs2_2nd_OR, No_Act_Lo_1vs2_2nd) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_Lo_1vs2_2nd_OR = ifelse(No_Act_Lo_1vs2_2nd>=35,35,No_Act_Lo_1vs2_2nd)) %>%
  as.data.frame()

## No_Act_Ph_2M_OR
ssDataOR %>% group_by(No_Act_Ph_2M) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_Ph_2M_OR = ifelse(No_Act_Ph_2M>=26,26,No_Act_Ph_2M)) %>%
  select(No_Act_Ph_2M_OR, No_Act_Ph_2M) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_Ph_2M_OR = ifelse(No_Act_Ph_2M>=26,26,No_Act_Ph_2M)) %>%
  as.data.frame()

## No_Act_WOF_10M_OR
ssDataOR %>% group_by(No_Act_WOF_10M) %>% summarise(count = n()) %>% View()
ssDataOR %>% mutate(No_Act_WOF_10M_OR = ifelse(No_Act_WOF_10M<0,0,No_Act_WOF_10M)) %>%
  select(No_Act_WOF_10M_OR, No_Act_WOF_10M) %>% View()

ssDataOR = ssDataOR %>% mutate(No_Act_WOF_10M_OR = ifelse(No_Act_WOF_10M<0,0,No_Act_WOF_10M)) %>%
  as.data.frame()

## No_List_Off_1M_OR
ssDataOR %>% group_by(No_List_Off_1M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$No_List_Off_1M)
ssDataOR %>% mutate(No_List_Off_1M_OR = ifelse(No_List_Off_1M>=20,20,No_List_Off_1M)) %>%
  select(No_List_Off_1M_OR, No_List_Off_1M) %>% View()

ssDataOR = ssDataOR %>% mutate(No_List_Off_1M_OR = ifelse(No_List_Off_1M>=20,20,No_List_Off_1M)) %>%
  as.data.frame()

## No_List_STST_3M_last

ssDataOR$No_List_STST_3M_last = ssDataOR$No_List_STST_6M - 
  ifelse(is.na(ssDataOR$No_List_STST_3M),0,ssDataOR$No_List_STST_3M)

ssDataOR %>% select(No_List_STST_3M_last, No_List_STST_6M, No_List_STST_3M) %>% View()

ssDataOR %>% group_by(No_List_STST_3M_last) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$No_List_STST_3M_last)

## No_List_T7_1M_OR
ssDataOR %>% group_by(No_List_T7_1M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$No_List_T7_1M)
ssDataOR %>% mutate(No_List_T7_1M_OR = ifelse(No_List_T7_1M>=178,178,No_List_T7_1M)) %>%
  select(No_List_T7_1M_OR, No_List_T7_1M) %>% View()

ssDataOR = ssDataOR %>% mutate(No_List_T7_1M_OR = ifelse(No_List_T7_1M>=178,178,No_List_T7_1M)) %>%
  as.data.frame()

## Avg_OAF_6M_OR
ssDataOR %>% group_by(Avg_OAF_6M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Avg_OAF_6M)
ssDataOR %>% mutate(Avg_OAF_6M_OR = ifelse(Avg_OAF_6M>=100,100,Avg_OAF_6M)) %>%
  select(Avg_OAF_6M_OR, Avg_OAF_6M) %>% View()

ssDataOR = ssDataOR %>% mutate(Avg_OAF_6M_OR = ifelse(Avg_OAF_6M>=100,100,Avg_OAF_6M)) %>%
  as.data.frame()

## AVG_Renew_p_Posting_OR
ssDataOR %>% group_by(AVG_Renew_p_Posting) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$AVG_Renew_p_Posting)
ssDataOR %>% mutate(AVG_Renew_p_Posting_OR = ifelse(AVG_Renew_p_Posting>=2,2,AVG_Renew_p_Posting)) %>%
  select(AVG_Renew_p_Posting_OR, AVG_Renew_p_Posting) %>% View()

ssDataOR = ssDataOR %>% mutate(AVG_Renew_p_Posting_OR = ifelse(AVG_Renew_p_Posting>=2,2,AVG_Renew_p_Posting)) %>%
  as.data.frame()

## AVG_STF_12M_OR
ssDataOR %>% group_by(AVG_STF_12M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$AVG_STF_12M)
ssDataOR %>% mutate(AVG_STF_12M_OR = ifelse(AVG_STF_12M>=33,33,AVG_STF_12M)) %>%
  select(AVG_STF_12M_OR, AVG_STF_12M) %>% View()

ssDataOR = ssDataOR %>% mutate(AVG_STF_12M_OR = ifelse(AVG_STF_12M>=33,33,AVG_STF_12M)) %>%
  as.data.frame()

## AVG_STF_6M_OR
ssDataOR %>% group_by(AVG_STF_6M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$AVG_STF_6M)
ssDataOR %>% mutate(AVG_STF_6M_OR = ifelse(AVG_STF_6M>=24,24,AVG_STF_6M)) %>%
  select(AVG_STF_6M_OR, AVG_STF_6M) %>% View()

ssDataOR = ssDataOR %>% mutate(AVG_STF_6M_OR = ifelse(AVG_STF_6M>=24,24,AVG_STF_6M)) %>%
  as.data.frame()

## Avg_Views_12M_OR
ssDataOR %>% group_by(Avg_Views_12M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Avg_Views_12M)
ssDataOR %>% mutate(Avg_Views_12M_OR = ifelse(Avg_Views_12M>=2309,2309,Avg_Views_12M)) %>%
  select(Avg_Views_12M_OR, Avg_Views_12M) %>% View()

ssDataOR = ssDataOR %>% mutate(Avg_Views_12M_OR = ifelse(Avg_Views_12M>=2309,2309,Avg_Views_12M)) %>%
  as.data.frame()

## Avg_Views_6M_OR
ssDataOR %>% group_by(Avg_Views_6M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Avg_Views_1M)
ssDataOR %>% mutate(Avg_Views_6M_OR = ifelse(Avg_Views_6M>=2282,2282,Avg_Views_6M)) %>%
  select(Avg_Views_6M_OR, Avg_Views_6M) %>% View()

ssDataOR = ssDataOR %>% mutate(Avg_Views_6M_OR = ifelse(Avg_Views_6M>=2282,2282,Avg_Views_6M)) %>%
  as.data.frame()

## Cre_Val_5M_OR
ssDataOR %>% group_by(Cre_Val_5M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Cre_Val_5M)
ssDataOR %>% mutate(Cre_Val_5M_OR = ifelse(Cre_Val_5M <= -116,-116,Cre_Val_5M)) %>%
  select(Cre_Val_5M_OR, Cre_Val_5M) %>% View()

ssDataOR = ssDataOR %>% mutate(Cre_Val_5M_OR = ifelse(Cre_Val_5M <= -116,-116,Cre_Val_5M)) %>%
  as.data.frame()

## Inv_Val_6M_last_OR
ssDataOR %>% group_by(Inv_Val_6M_last) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Inv_Val_6M_last)
ssDataOR %>% mutate(Inv_Val_6M_last_OR = ifelse(Inv_Val_6M_last <= 0,0,Inv_Val_6M_last)) %>%
  select(Inv_Val_6M_last_OR, Inv_Val_6M_last) %>% View()

ssDataOR = ssDataOR %>% mutate(Inv_Val_6M_last_OR = ifelse(Inv_Val_6M_last <= 0,0,Inv_Val_6M_last)) %>%
  as.data.frame()

## Inv_Val_1M
ssDataOR %>% group_by(Inv_Val_1M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Inv_Val_1M)
ssDataOR %>% mutate(Inv_Val_1M_OR = ifelse(Inv_Val_1M <= 0,0,Inv_Val_1M)) %>%
  select(Inv_Val_1M_OR, Inv_Val_1M) %>% View()

ssDataOR = ssDataOR %>% mutate(Inv_Val_1M_OR = ifelse(Inv_Val_1M <= 0,0,Inv_Val_1M)) %>%
  as.data.frame()

## Inv_Val_2M_OR
ssDataOR %>% group_by(Inv_Val_2M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$Inv_Val_2M)
ssDataOR %>% mutate(Inv_Val_2M_OR = ifelse(Inv_Val_2M <= 0,0,Inv_Val_2M)) %>%
  select(Inv_Val_2M_OR, Inv_Val_2M) %>% View()

ssDataOR = ssDataOR %>% mutate(Inv_Val_2M_OR = ifelse(Inv_Val_2M <= 0,0,Inv_Val_2M)) %>%
  as.data.frame()

## List_FB_off_6M_OR
ssDataOR %>% group_by(List_FB_off_6M) %>% summarise(count = n()) %>% View()
boxplot(ssDataOR$List_FB_off_6M)
ssDataOR %>% mutate(List_FB_off_6M_OR = ifelse(List_FB_off_6M >= 250,250,List_FB_off_6M)) %>%
  select(List_FB_off_6M_OR, List_FB_off_6M) %>% View()

ssDataOR = ssDataOR %>% mutate(List_FB_off_6M_OR = ifelse(List_FB_off_6M >= 250,250,List_FB_off_6M)) %>%
  as.data.frame()

write.csv(ssDataOR, file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataORemoved.csv"
          , row.names = FALSE)

ssDataORemoved = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataORemoved.csv"
                          , na.strings = c("", " ", "NA", "na",NA))


View(ssDataORemoved)




############################ Modelling ############################

ssDataModel = ssDataORemoved %>% select(ID, Target_Sold, Target_Sales_OR, Var_04, No_Act_1M_no_Ph_OR
                                        , No_Act_2M_no_Ph_OR, No_Act_3M_all_OR, No_Act_Lo_1M_OR
                                        , No_Act_Lo_1vs2_2nd_OR, No_Act_Ph_1M, No_Act_Ph_2M_OR
                                        , No_Act_WOF_1M, No_Act_WOF_1vs2_2nd, No_Act_WOF_10M_OR
                                        , No_list_FB_6M, No_list_IND_24M, No_List_Off_1M_OR
                                        , No_List_On_24M, No_List_STST_3M, No_List_STST_3M_last
                                        , No_List_T7_1M_OR, Avg_OAF_6M_OR, AVG_Refresh_p_Posting
                                        , AVG_Renew_p_Posting_OR, AVG_Share_Online, AVG_STF_12M_OR
                                        , AVG_STF_1M, AVG_STF_6M_OR, Avg_Views_12M_OR
                                        , Avg_Views_1M, Avg_Views_6M_OR, Count_SIC_3
                                        , Cre_Val_6M_last, Cre_Val_1M, Cre_Val_5M_OR
                                        , Inv_Days_6M_last, Inv_Days_1M, Inv_Days_5M
                                        , Inv_Val_6M_last_OR, Inv_Val_1M_OR, Inv_Val_2M_OR
                                        , List_FB_HOM_1M, List_FB_off_6M_OR
                                        , List_STS_FB_1M, List_STS_Index_12M
                                        , List_STS_Index_1M, List_STS_Index_3M
                                        , List_STS_off_24M, List_STS_T4_12M
                                        , MISC_Days_Since_Last_act_noffer
                                        , MISC_Days_Since_Last_Offer
                                        , Offer_days_1M
                                        , Offer_days_23M)

View(ssDataModel)

# one-hot for Var_04
ssDataModelTemp = ssDataModel
ssDataModelTemp$Var_04 = factor(ssDataModel$Var_04)
encoder = onehot(ssDataModelTemp[,c("Var_04"), drop = FALSE], max_levels = 100000)
oneHotPredict = predict(encoder, ssDataModelTemp[,c("Var_04"), drop = FALSE])
dfOneHot = data.frame(oneHotPredict)
colnames(dfOneHot) = as.character(dimnames(oneHotPredict)[[2]])

ssDataModel = ssDataModel %>% cbind(dfOneHot) %>% as.data.frame()
nrow(ssDataModel)
View(ssDataModel)

#### NA operations ####

# Target_Sales_ORNA
ssDataModel %>% mutate(Target_Sales_ORNA = ifelse(is.na(Target_Sales_OR),0,Target_Sales_OR)) %>%
  select(Target_Sales_ORNA, Target_Sales_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Target_Sales_ORNA = ifelse(is.na(Target_Sales_OR),0,Target_Sales_OR)) %>%
  as.data.frame()

# No_Act_1M_no_Ph_ORNA
ssDataModel %>% mutate(No_Act_1M_no_Ph_ORNA = ifelse(is.na(No_Act_1M_no_Ph_OR),0,No_Act_1M_no_Ph_OR)) %>%
  select(No_Act_1M_no_Ph_ORNA, No_Act_1M_no_Ph_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_1M_no_Ph_ORNA = ifelse(is.na(No_Act_1M_no_Ph_OR),0,No_Act_1M_no_Ph_OR)) %>%
  as.data.frame()

# No_Act_2M_no_Ph_ORNA
ssDataModel %>% mutate(No_Act_2M_no_Ph_ORNA = ifelse(is.na(No_Act_2M_no_Ph_OR),0,No_Act_2M_no_Ph_OR)) %>%
  select(No_Act_2M_no_Ph_ORNA, No_Act_2M_no_Ph_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_2M_no_Ph_ORNA = ifelse(is.na(No_Act_2M_no_Ph_OR),0,No_Act_2M_no_Ph_OR)) %>%
  as.data.frame()

# No_Act_3M_all_ORNA
ssDataModel %>% mutate(No_Act_3M_all_ORNA = ifelse(is.na(No_Act_3M_all_OR),0,No_Act_3M_all_OR)) %>%
  select(No_Act_3M_all_ORNA, No_Act_3M_all_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_3M_all_ORNA = ifelse(is.na(No_Act_3M_all_OR),0,No_Act_3M_all_OR)) %>%
  as.data.frame()

# No_Act_Lo_1M_ORNA
ssDataModel %>% mutate(No_Act_Lo_1M_ORNA = ifelse(is.na(No_Act_Lo_1M_OR),0,No_Act_Lo_1M_OR)) %>%
  select(No_Act_Lo_1M_ORNA, No_Act_Lo_1M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_Lo_1M_ORNA = ifelse(is.na(No_Act_Lo_1M_OR),0,No_Act_Lo_1M_OR)) %>%
  as.data.frame()

# No_Act_Lo_1vs2_2nd_ORNA
ssDataModel %>% mutate(No_Act_Lo_1vs2_2nd_ORNA = ifelse(is.na(No_Act_Lo_1vs2_2nd_OR),0,No_Act_Lo_1vs2_2nd_OR)) %>%
  select(No_Act_Lo_1vs2_2nd_ORNA, No_Act_Lo_1vs2_2nd_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_Lo_1vs2_2nd_ORNA = ifelse(is.na(No_Act_Lo_1vs2_2nd_OR),0,No_Act_Lo_1vs2_2nd_OR)) %>%
  as.data.frame()

# No_Act_Ph_1MNA
ssDataModel %>% mutate(No_Act_Ph_1MNA = ifelse(is.na(No_Act_Ph_1M),0,No_Act_Ph_1M)) %>%
  select(No_Act_Ph_1MNA, No_Act_Ph_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_Ph_1MNA = ifelse(is.na(No_Act_Ph_1M),0,No_Act_Ph_1M)) %>%
  as.data.frame()

# No_Act_Ph_2M_ORNA
ssDataModel %>% mutate(No_Act_Ph_2M_ORNA = ifelse(is.na(No_Act_Ph_2M_OR),0,No_Act_Ph_2M_OR)) %>%
  select(No_Act_Ph_2M_ORNA, No_Act_Ph_2M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_Ph_2M_ORNA = ifelse(is.na(No_Act_Ph_2M_OR),0,No_Act_Ph_2M_OR)) %>%
  as.data.frame()

# No_Act_WOF_1MNA
ssDataModel %>% mutate(No_Act_WOF_1MNA = ifelse(is.na(No_Act_WOF_1M),0,No_Act_WOF_1M)) %>%
  select(No_Act_WOF_1MNA, No_Act_WOF_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_WOF_1MNA = ifelse(is.na(No_Act_WOF_1M),0,No_Act_WOF_1M)) %>%
  as.data.frame()

# No_Act_WOF_1vs2_2ndNA
ssDataModel %>% mutate(No_Act_WOF_1vs2_2ndNA = ifelse(is.na(No_Act_WOF_1vs2_2nd),0,No_Act_WOF_1vs2_2nd)) %>%
  select(No_Act_WOF_1vs2_2ndNA, No_Act_WOF_1vs2_2nd) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_WOF_1vs2_2ndNA = ifelse(is.na(No_Act_WOF_1vs2_2nd),0,No_Act_WOF_1vs2_2nd)) %>%
  as.data.frame()

# No_Act_WOF_10M_ORNA
ssDataModel %>% mutate(No_Act_WOF_10M_ORNA = ifelse(is.na(No_Act_WOF_10M_OR),0,No_Act_WOF_10M_OR)) %>%
  select(No_Act_WOF_10M_ORNA, No_Act_WOF_10M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_Act_WOF_10M_ORNA = ifelse(is.na(No_Act_WOF_10M_OR),0,No_Act_WOF_10M_OR)) %>%
  as.data.frame()

# No_list_IND_24MNA
ssDataModel %>% mutate(No_list_IND_24MNA = ifelse(is.na(No_list_IND_24M),0,No_list_IND_24M)) %>%
  select(No_list_IND_24MNA, No_list_IND_24M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_list_IND_24MNA = ifelse(is.na(No_list_IND_24M),0,No_list_IND_24M)) %>%
  as.data.frame()

# No_List_Off_1M_ORNA
ssDataModel %>% mutate(No_List_Off_1M_ORNA = ifelse(is.na(No_List_Off_1M_OR),0,No_List_Off_1M_OR)) %>%
  select(No_List_Off_1M_ORNA, No_List_Off_1M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_List_Off_1M_ORNA = ifelse(is.na(No_List_Off_1M_OR),0,No_List_Off_1M_OR)) %>%
  as.data.frame()

# No_List_On_24MNA
ssDataModel %>% mutate(No_List_On_24MNA = ifelse(is.na(No_List_On_24M),0,No_List_On_24M)) %>%
  select(No_List_On_24MNA, No_List_On_24M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_List_On_24MNA = ifelse(is.na(No_List_On_24M),0,No_List_On_24M)) %>%
  as.data.frame()

# No_List_STST_3MNA
ssDataModel %>% mutate(No_List_STST_3MNA = ifelse(is.na(No_List_STST_3M),0,No_List_STST_3M)) %>%
  select(No_List_STST_3MNA, No_List_STST_3M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_List_STST_3MNA = ifelse(is.na(No_List_STST_3M),0,No_List_STST_3M)) %>%
  as.data.frame()

# No_List_STST_3M_lastNA
ssDataModel %>% mutate(No_List_STST_3M_lastNA = ifelse(is.na(No_List_STST_3M_last),0,No_List_STST_3M_last)) %>%
  select(No_List_STST_3M_lastNA, No_List_STST_3M_last) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_List_STST_3M_lastNA = ifelse(is.na(No_List_STST_3M_last),0,No_List_STST_3M_last)) %>%
  as.data.frame()

# No_List_T7_1M_ORNA
ssDataModel %>% mutate(No_List_T7_1M_ORNA = ifelse(is.na(No_List_T7_1M_OR),0,No_List_T7_1M_OR)) %>%
  select(No_List_T7_1M_ORNA, No_List_T7_1M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(No_List_T7_1M_ORNA = ifelse(is.na(No_List_T7_1M_OR),0,No_List_T7_1M_OR)) %>%
  as.data.frame()

# Count_SIC_3NA
ssDataModel %>% mutate(Count_SIC_3NA = ifelse(is.na(Count_SIC_3),0,Count_SIC_3)) %>%
  select(Count_SIC_3NA, Count_SIC_3) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Count_SIC_3NA = ifelse(is.na(Count_SIC_3),0,Count_SIC_3)) %>%
  as.data.frame()

# Cre_Val_6M_lastNA
ssDataModel %>% mutate(Cre_Val_6M_lastNA = ifelse(is.na(Cre_Val_6M_last),0,Cre_Val_6M_last)) %>%
  select(Cre_Val_6M_lastNA, Cre_Val_6M_last) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Cre_Val_6M_lastNA = ifelse(is.na(Cre_Val_6M_last),0,Cre_Val_6M_last)) %>%
  as.data.frame()

# Cre_Val_1MNA
ssDataModel %>% mutate(Cre_Val_1MNA = ifelse(is.na(Cre_Val_1M),0,Cre_Val_1M)) %>%
  select(Cre_Val_1MNA, Cre_Val_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Cre_Val_1MNA = ifelse(is.na(Cre_Val_1M),0,Cre_Val_1M)) %>%
  as.data.frame()

# Cre_Val_5M_ORNA
ssDataModel %>% mutate(Cre_Val_5M_ORNA = ifelse(is.na(Cre_Val_5M_OR),0,Cre_Val_5M_OR)) %>%
  select(Cre_Val_5M_ORNA, Cre_Val_5M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Cre_Val_5M_ORNA = ifelse(is.na(Cre_Val_5M_OR),0,Cre_Val_5M_OR)) %>%
  as.data.frame()

# Inv_Days_6M_lastNA
ssDataModel %>% mutate(Inv_Days_6M_lastNA = ifelse(is.na(Inv_Days_6M_last),0,Inv_Days_6M_last)) %>%
  select(Inv_Days_6M_lastNA, Inv_Days_6M_last) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Days_6M_lastNA = ifelse(is.na(Inv_Days_6M_last),0,Inv_Days_6M_last)) %>%
  as.data.frame()

# Inv_Days_1MNA
ssDataModel %>% mutate(Inv_Days_1MNA = ifelse(is.na(Inv_Days_1M),0,Inv_Days_1M)) %>%
  select(Inv_Days_1MNA, Inv_Days_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Days_1MNA = ifelse(is.na(Inv_Days_1M),0,Inv_Days_1M)) %>%
  as.data.frame()

# Inv_Days_5MNA
ssDataModel %>% mutate(Inv_Days_5MNA = ifelse(is.na(Inv_Days_5M),0,Inv_Days_5M)) %>%
  select(Inv_Days_5MNA, Inv_Days_5M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Days_5MNA = ifelse(is.na(Inv_Days_5M),0,Inv_Days_5M)) %>%
  as.data.frame()

# Inv_Val_6M_last_ORNA
ssDataModel %>% mutate(Inv_Val_6M_last_ORNA = ifelse(is.na(Inv_Val_6M_last_OR),0,Inv_Val_6M_last_OR)) %>%
  select(Inv_Val_6M_last_ORNA, Inv_Val_6M_last_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Val_6M_last_ORNA = ifelse(is.na(Inv_Val_6M_last_OR),0,Inv_Val_6M_last_OR)) %>%
  as.data.frame()

# Inv_Val_1M_ORNA
ssDataModel %>% mutate(Inv_Val_1M_ORNA = ifelse(is.na(Inv_Val_1M_OR),0,Inv_Val_1M_OR)) %>%
  select(Inv_Val_1M_ORNA, Inv_Val_1M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Val_1M_ORNA = ifelse(is.na(Inv_Val_1M_OR),0,Inv_Val_1M_OR)) %>%
  as.data.frame()

# Inv_Val_2M_ORNA
ssDataModel %>% mutate(Inv_Val_2M_ORNA = ifelse(is.na(Inv_Val_2M_OR),0,Inv_Val_2M_OR)) %>%
  select(Inv_Val_2M_ORNA, Inv_Val_2M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Inv_Val_2M_ORNA = ifelse(is.na(Inv_Val_2M_OR),0,Inv_Val_2M_OR)) %>%
  as.data.frame()

# MISC_Days_Since_Last_act_nofferNA
ssDataModel %>% mutate(MISC_Days_Since_Last_act_nofferNA = ifelse(is.na(MISC_Days_Since_Last_act_noffer),0,MISC_Days_Since_Last_act_noffer)) %>%
  select(MISC_Days_Since_Last_act_nofferNA, MISC_Days_Since_Last_act_noffer) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(MISC_Days_Since_Last_act_nofferNA = ifelse(is.na(MISC_Days_Since_Last_act_noffer),0,MISC_Days_Since_Last_act_noffer)) %>%
  as.data.frame()

# MISC_Days_Since_Last_OfferNA
ssDataModel %>% mutate(MISC_Days_Since_Last_OfferNA = ifelse(is.na(MISC_Days_Since_Last_Offer),0,MISC_Days_Since_Last_Offer)) %>%
  select(MISC_Days_Since_Last_OfferNA, MISC_Days_Since_Last_Offer) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(MISC_Days_Since_Last_OfferNA = ifelse(is.na(MISC_Days_Since_Last_Offer),0,MISC_Days_Since_Last_Offer)) %>%
  as.data.frame()

# Offer_days_1MNA
ssDataModel %>% mutate(Offer_days_1MNA = ifelse(is.na(Offer_days_1M),0,Offer_days_1M)) %>%
  select(Offer_days_1MNA, Offer_days_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Offer_days_1MNA = ifelse(is.na(Offer_days_1M),0,Offer_days_1M)) %>%
  as.data.frame()

# Offer_days_23MNA
ssDataModel %>% mutate(Offer_days_23MNA = ifelse(is.na(Offer_days_23M),0,Offer_days_23M)) %>%
  select(Offer_days_23MNA, Offer_days_23M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Offer_days_23MNA = ifelse(is.na(Offer_days_23M),0,Offer_days_23M)) %>%
  as.data.frame()

# Avg_OAF_6M_ORNA
Mode <- function(x, na.rm = TRUE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

Mode(ssDataModel$Avg_OAF_6M_OR, na.rm = TRUE) 
mean(ssDataModel$Avg_OAF_6M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(Avg_OAF_6M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(Avg_OAF_6M_ORNA = ifelse(is.na(Avg_OAF_6M_OR),15.65651,Avg_OAF_6M_OR)) %>%
  select(Avg_OAF_6M_ORNA, Avg_OAF_6M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Avg_OAF_6M_ORNA = ifelse(is.na(Avg_OAF_6M_OR),15.65651,Avg_OAF_6M_OR)) %>%
  as.data.frame()

# AVG_Refresh_p_PostingNA
Mode(ssDataModel$AVG_Refresh_p_Posting, na.rm = TRUE) 
mean(ssDataModel$AVG_Refresh_p_Posting, na.rm = TRUE) 

ssDataModel %>% group_by(AVG_Refresh_p_Posting) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_Refresh_p_PostingNA = ifelse(is.na(AVG_Refresh_p_Posting),1,AVG_Refresh_p_Posting)) %>%
  select(AVG_Refresh_p_PostingNA, AVG_Refresh_p_Posting) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_Refresh_p_PostingNA = ifelse(is.na(AVG_Refresh_p_Posting),1,AVG_Refresh_p_Posting)) %>%
  as.data.frame()

# AVG_Renew_p_Posting_ORNA
Mode(ssDataModel$AVG_Renew_p_Posting_OR, na.rm = TRUE) 
mean(ssDataModel$AVG_Renew_p_Posting_OR, na.rm = TRUE) # 0.1740891

ssDataModel %>% group_by(AVG_Renew_p_Posting_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_Renew_p_Posting_ORNA = ifelse(is.na(AVG_Renew_p_Posting_OR),0.1740891,AVG_Renew_p_Posting_OR)) %>%
  select(AVG_Renew_p_Posting_ORNA, AVG_Renew_p_Posting_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_Renew_p_Posting_ORNA = ifelse(is.na(AVG_Renew_p_Posting_OR),0.1740891,AVG_Renew_p_Posting_OR)) %>%
  as.data.frame()

# AVG_Share_OnlineNA
Mode(ssDataModel$AVG_Share_Online, na.rm = TRUE) 
mean(ssDataModel$AVG_Share_Online, na.rm = TRUE) 

ssDataModel %>% group_by(AVG_Share_Online) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_Share_OnlineNA = ifelse(is.na(AVG_Share_Online),0.824743,AVG_Share_Online)) %>%
  select(AVG_Share_OnlineNA, AVG_Share_Online) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_Share_OnlineNA = ifelse(is.na(AVG_Share_Online),0.824743,AVG_Share_Online)) %>%
  as.data.frame()

# AVG_STF_12M_ORNA
Mode(ssDataModel$AVG_STF_12M_OR, na.rm = TRUE) 
mean(ssDataModel$AVG_STF_12M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(AVG_STF_12M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_STF_12M_ORNA = ifelse(is.na(AVG_STF_12M_OR),3.311466,AVG_STF_12M_OR)) %>%
  select(AVG_STF_12M_ORNA, AVG_STF_12M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_STF_12M_ORNA = ifelse(is.na(AVG_STF_12M_OR),3.311466,AVG_STF_12M_OR)) %>%
  as.data.frame()

# AVG_STF_1MNA
Mode(ssDataModel$AVG_STF_1M, na.rm = TRUE) 
mean(ssDataModel$AVG_STF_1M, na.rm = TRUE) 

ssDataModel %>% group_by(AVG_STF_1M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_STF_1MNA = ifelse(is.na(AVG_STF_1M),1.144531,AVG_STF_1M)) %>%
  select(AVG_STF_1MNA, AVG_STF_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_STF_1MNA = ifelse(is.na(AVG_STF_1M),1.144531,AVG_STF_1M)) %>%
  as.data.frame()

# AVG_STF_6M_ORNA
Mode(ssDataModel$AVG_STF_6M_OR, na.rm = TRUE) 
mean(ssDataModel$AVG_STF_6M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(AVG_STF_6M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(AVG_STF_6M_ORNA = ifelse(is.na(AVG_STF_6M_OR),2.656348,AVG_STF_6M_OR)) %>%
  select(AVG_STF_6M_ORNA, AVG_STF_6M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(AVG_STF_6M_ORNA = ifelse(is.na(AVG_STF_6M_OR),2.656348,AVG_STF_6M_OR)) %>%
  as.data.frame()

# Avg_Views_12M_ORNA
Mode(ssDataModel$Avg_Views_12M_OR, na.rm = TRUE) 
mean(ssDataModel$Avg_Views_12M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(Avg_Views_12M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(Avg_Views_12M_ORNA = ifelse(is.na(Avg_Views_12M_OR),418.2829,Avg_Views_12M_OR)) %>%
  select(Avg_Views_12M_ORNA, Avg_Views_12M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Avg_Views_12M_ORNA = ifelse(is.na(Avg_Views_12M_OR),418.2829,Avg_Views_12M_OR)) %>%
  as.data.frame()

# Avg_Views_1MNA
Mode(ssDataModel$Avg_Views_1M, na.rm = TRUE) 
mean(ssDataModel$Avg_Views_1M, na.rm = TRUE) 

ssDataModel %>% group_by(Avg_Views_1M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(Avg_Views_1MNA = ifelse(is.na(Avg_Views_1M),148.8115,Avg_Views_1M)) %>%
  select(Avg_Views_1MNA, Avg_Views_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Avg_Views_1MNA = ifelse(is.na(Avg_Views_1M),148.8115,Avg_Views_1M)) %>%
  as.data.frame()

# Avg_Views_6M_ORNA
Mode(ssDataModel$Avg_Views_6M_OR, na.rm = TRUE) 
mean(ssDataModel$Avg_Views_6M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(Avg_Views_6M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(Avg_Views_6M_ORNA = ifelse(is.na(Avg_Views_6M_OR),339.5198,Avg_Views_6M_OR)) %>%
  select(Avg_Views_6M_ORNA, Avg_Views_6M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(Avg_Views_6M_ORNA = ifelse(is.na(Avg_Views_6M_OR),339.5198,Avg_Views_6M_OR)) %>%
  as.data.frame()

# List_FB_HOM_1M
Mode(ssDataModel$List_FB_HOM_1M, na.rm = TRUE) 
mean(ssDataModel$List_FB_HOM_1M, na.rm = TRUE) 

ssDataModel %>% group_by(List_FB_HOM_1M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_FB_HOM_1MNA = ifelse(is.na(List_FB_HOM_1M),3.852415,List_FB_HOM_1M)) %>%
  select(List_FB_HOM_1MNA, List_FB_HOM_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_FB_HOM_1MNA = ifelse(is.na(List_FB_HOM_1M),3.852415,List_FB_HOM_1M)) %>%
  as.data.frame()

# List_FB_off_6M_ORNA
Mode(ssDataModel$List_FB_off_6M_OR, na.rm = TRUE) 
mean(ssDataModel$List_FB_off_6M_OR, na.rm = TRUE) 

ssDataModel %>% group_by(List_FB_off_6M_OR) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_FB_off_6M_ORNA = ifelse(is.na(List_FB_off_6M_OR),19.02629,List_FB_off_6M_OR)) %>%
  select(List_FB_off_6M_ORNA, List_FB_off_6M_OR) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_FB_off_6M_ORNA = ifelse(is.na(List_FB_off_6M_OR),19.02629,List_FB_off_6M_OR)) %>%
  as.data.frame()

# List_STS_FB_1MNA
Mode(ssDataModel$List_STS_FB_1M, na.rm = TRUE) 
mean(ssDataModel$List_STS_FB_1M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_FB_1M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_FB_1MNA = ifelse(is.na(List_STS_FB_1M),0.8590279,List_STS_FB_1M)) %>%
  select(List_STS_FB_1MNA, List_STS_FB_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_FB_1MNA = ifelse(is.na(List_STS_FB_1M),0.8590279,List_STS_FB_1M)) %>%
  as.data.frame()

# List_STS_Index_12MNA
Mode(ssDataModel$List_STS_Index_12M, na.rm = TRUE) 
mean(ssDataModel$List_STS_Index_12M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_Index_12M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_Index_12MNA = ifelse(is.na(List_STS_Index_12M),0.2364922,List_STS_Index_12M)) %>%
  select(List_STS_Index_12MNA, List_STS_Index_12M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_Index_12MNA = ifelse(is.na(List_STS_Index_12M),0.2364922,List_STS_Index_12M)) %>%
  as.data.frame()

# List_STS_Index_1MNA
Mode(ssDataModel$List_STS_Index_1M, na.rm = TRUE) 
mean(ssDataModel$List_STS_Index_1M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_Index_1M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_Index_1MNA = ifelse(is.na(List_STS_Index_1M),0.395431,List_STS_Index_1M)) %>%
  select(List_STS_Index_1MNA, List_STS_Index_1M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_Index_1MNA = ifelse(is.na(List_STS_Index_1M),0.395431,List_STS_Index_1M)) %>%
  as.data.frame()

# List_STS_Index_3MNA
Mode(ssDataModel$List_STS_Index_3M, na.rm = TRUE) 
mean(ssDataModel$List_STS_Index_3M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_Index_3M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_Index_3MNA = ifelse(is.na(List_STS_Index_3M),0.3280768,List_STS_Index_3M)) %>%
  select(List_STS_Index_3MNA, List_STS_Index_3M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_Index_3MNA = ifelse(is.na(List_STS_Index_3M),0.3280768,List_STS_Index_3M)) %>%
  as.data.frame()

# List_STS_off_24MNA
Mode(ssDataModel$List_STS_off_24M, na.rm = TRUE) 
mean(ssDataModel$List_STS_off_24M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_off_24M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_off_24MNA = ifelse(is.na(List_STS_off_24M),20.34903,List_STS_off_24M)) %>%
  select(List_STS_off_24MNA, List_STS_off_24M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_off_24MNA = ifelse(is.na(List_STS_off_24M),20.34903,List_STS_off_24M)) %>%
  as.data.frame()

# List_STS_T4_12MNA
Mode(ssDataModel$List_STS_T4_12M, na.rm = TRUE) 
mean(ssDataModel$List_STS_T4_12M, na.rm = TRUE) 

ssDataModel %>% group_by(List_STS_T4_12M) %>% summarise(count = n()) %>% View()

ssDataModel %>% mutate(List_STS_T4_12MNA = ifelse(is.na(List_STS_T4_12M),5.225781,List_STS_T4_12M)) %>%
  select(List_STS_T4_12MNA, List_STS_T4_12M) %>% View()

ssDataModel = ssDataModel %>% 
  mutate(List_STS_T4_12MNA = ifelse(is.na(List_STS_T4_12M),5.225781,List_STS_T4_12M)) %>%
  as.data.frame()

write.csv(ssDataModel, file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataModel.csv"
          , row.names = FALSE)

ssDataModel = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataModel.csv"
                         , na.strings = c("", " ", "NA", "na",NA))


ssDataModelNA = ssDataModel %>% select(ID,
                       Target_Sold,
                       Target_Sales_ORNA,
                       Var_04.0,
                       Var_04.1,
                       Var_04.3,
                       Var_04.6,
                       Var_04.12,
                       Var_04.23,
                       Var_04.24,
                       No_Act_1M_no_Ph_ORNA,
                       No_Act_2M_no_Ph_ORNA,
                       No_Act_3M_all_ORNA,
                       No_Act_Lo_1M_ORNA,
                       No_Act_Lo_1vs2_2nd_ORNA,
                       No_Act_Ph_1MNA,
                       No_Act_Ph_2M_ORNA,
                       No_Act_WOF_1MNA,
                       No_Act_WOF_1vs2_2ndNA,
                       No_Act_WOF_10M_ORNA,
                       No_list_FB_6M,
                       No_list_IND_24MNA,
                       No_List_Off_1M_ORNA,
                       No_List_On_24MNA,
                       No_List_STST_3MNA,
                       No_List_STST_3M_lastNA,
                       No_List_T7_1M_ORNA,
                       Avg_OAF_6M_ORNA,
                       AVG_Refresh_p_PostingNA,
                       AVG_Renew_p_Posting_ORNA,
                       AVG_Share_OnlineNA,
                       AVG_STF_12M_ORNA,
                       AVG_STF_1MNA,
                       AVG_STF_6M_ORNA,
                       Avg_Views_12M_ORNA,
                       Avg_Views_1MNA,
                       Avg_Views_6M_ORNA,
                       Count_SIC_3NA,
                       Cre_Val_6M_lastNA,
                       Cre_Val_1MNA,
                       Cre_Val_5M_ORNA,
                       Inv_Days_6M_lastNA,
                       Inv_Days_1MNA,
                       Inv_Days_5MNA,
                       Inv_Val_6M_last_ORNA,
                       Inv_Val_1M_ORNA,
                       Inv_Val_2M_ORNA,
                       List_FB_HOM_1MNA,
                       List_FB_off_6M_ORNA,
                       List_STS_FB_1MNA,
                       List_STS_Index_12MNA,
                       List_STS_Index_1MNA,
                       List_STS_Index_3MNA,
                       List_STS_off_24MNA,
                       List_STS_T4_12MNA,
                       MISC_Days_Since_Last_act_nofferNA,
                       MISC_Days_Since_Last_OfferNA,
                       Offer_days_1MNA,
                       Offer_days_23MNA)

write.csv(ssDataModelNA, file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataModelNA.csv"
            , row.names = FALSE)

ssDataModelNA = read.csv(file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataModelNA.csv"
                         , na.strings = c("", " ", "NA", "na",NA))
View(ssDataModelNA)

#### XGBoost for sales value prediction ####

excludeColumns = c("ID","Target_Sold")
considerColumns = setdiff(colnames(ssDataModelNA), excludeColumns)

ssDataModelNARegression = ssDataModelNA %>% filter(Target_Sold==1) %>% as.data.frame()

trainXGB = ssDataModelNARegression[, considerColumns]
trainXGB$Target_Sales_ORNALog = log(trainXGB$Target_Sales_ORNA+.01)

# Train data split
intrain = caret :: createDataPartition(y=trainXGB$Target_Sales_ORNALog,p=0.8,list=FALSE)
trainData = trainXGB[intrain,]
testData = trainXGB[-intrain,]

xgbTree = xgboost(data = as.matrix(select(trainData, -Target_Sales_ORNA, -Target_Sales_ORNALog))
                  , label = trainData$Target_Sales_ORNALog
                  , max_depth = 6
                  , eta = .1
                  , gamma = 2
                  , nthread = 5
                  , nrounds = 500
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .8
                  , colsample_bytree = .6
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)

performXGB = predict(xgbTree, newdata = as.matrix(select(testData, -Target_Sales_ORNA, -Target_Sales_ORNALog)))
# Normal
performRMSEXGB = sqrt(mean((testData$Target_Sales_ORNA - performXGB)^2))
print(performRMSEXGB)
# Log transformation
performRMSEXGB = sqrt(mean((testData$Target_Sales_ORNALog - (performXGB))^2))
print(performRMSEXGB)

# Normal
xgbAnalyse = data.frame("real" = testData$Target_Sales_ORNA
                        , "predicted" = performXGB
                        #, "Date" = testData$Date
)
View(xgbAnalyse)
# Log transformation
xgbAnalyse = data.frame("real" = testData$rpc
                        , "predicted" = exp(performXGB)
                        #, "Date" = testData$Date
)





#### Hyperparameter tuning ####
set.seed(789)
# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = c(100, 200)
  , max_depth = c(5, 10, 15)
  , eta = c(.1, 0.05)
  , gamma = c(.05, 1, 2)
  , subsample = c(0.6, 0.8, 0.9)
  , colsample_bytree = c(0.6, 0.7, .9)
  , min_child_weight = c(1,2)
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv"
  , number = 3
  , allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
# using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(select(trainData, -Target_Sales_ORNA, -Target_Sales_ORNALog))
  , y = as.numeric(trainData$Target_Sales_ORNA)
  , trControl = xgb_trcontrol_1
  , tuneGrid = xgb_grid_1
  , method = "xgbTree"
  , metric = "RMSE"
)

write.csv(xgb_train_1$results, file = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/bestModel.csv")
xgb_train_1$bestTune
# nrounds max_depth  eta gamma colsample_bytree
# 100        15 0.05     2              0.9
# min_child_weight subsample
#            1       0.6

## Clear indication that the model can overfit if not restricted properly
# Need to restrict by max_depth, colsample_bytree and min_child_weight

gridSearchResult = xgb_train_1$results
min(gridSearchResult$RMSE)
gridSearchResult %>% filter(RMSE <= 5330) %>%View()


####################################################################
#### Final Model ####
set.seed(1000)
excludeColumns = c("ID","Target_Sold")
considerColumns = setdiff(colnames(ssDataModelNA), excludeColumns)
#considerColumns = c(considerColumns, "Var_04")

ssDataModelNARegression = ssDataModelNA %>% filter(Target_Sold==1) %>% as.data.frame()
# bring back the Var column

#ssDataModelNARegression = select(ssData,ID,Var_04) %>% 
 # inner_join(ssDataModelNARegression, by = c("ID")) %>% as.data.frame()

noScale = c()#"Var_04.0", "Var_04.1", "Var_04.3", "Var_04.6", "Var_04.12", "Var_04.23"
            #, "Var_04.24")
colWithScale = setdiff(considerColumns,noScale)
trainXGB = ssDataModelNARegression[, colWithScale]
trainXGB$Target_Sales_ORNALog = log(trainXGB$Target_Sales_ORNA)
# requireUnscale = scale(trainXGB, center = TRUE, scale = TRUE)
# trainXGB = data.frame(scale(trainXGB, center = TRUE, scale = TRUE))

intrain = caret :: createDataPartition(y=trainXGB$Target_Sales_ORNA,p=0.8,list=FALSE)
trainData = trainXGB[intrain,]
testData = trainXGB[-intrain,]

xgbTree = xgboost(data = as.matrix(select(trainData, -Target_Sales_ORNA, -Target_Sales_ORNALog))
                  , label = trainData$Target_Sales_ORNA
                  , max_depth = 3
                  , min_child_weight = 8
                  #, lambda = 2
                  #, lambda_bias = 1
                  , eta = .2
                  , gamma = 10
                  , nthread = 5
                  , nrounds = 250
                  , booster = 'gbtree' # gbtree vs gblinear
                  , subsample = .8
                  , colsample_bytree = .4
                  , objective = "reg:linear"
                  , eval_metric = "rmse"
                  , verbose = 1)
#lambda = 1, lambda_bias = 1,gamma = 2,
# 2798.634
performXGB = predict(xgbTree
                     , newdata = as.matrix(select(testData
                                                  , -Target_Sales_ORNA
                                                  , - Target_Sales_ORNALog)))
performXGB = data.frame(Target_Sales_ORNA = performXGB)
## Normal
performRMSEXGB = sqrt(mean((testData$Target_Sales_ORNA - performXGB$Target_Sales_ORNA)^2))
print(performRMSEXGB) # 4029.508

## Create data frame to plot
index = seq(1,nrow(testData),1)
xgbAnalyse2 = data.frame(index = index
                         ,"Real" = testData$Target_Sales_ORNA
                        , "Predicted" = (performXGB$Target_Sales_ORNA)
                        #, "Date" = testData$Date
)
View(xgbAnalyse2)

## plot the prediction
xgbAnalyse2Reshaped = melt(xgbAnalyse2, id="index")

xgbAnalyse2Reshaped = xgbAnalyse2Reshaped %>% mutate(Value = ifelse(value<0, 0, value)) %>% 
  select(index, variable, Value) %>% as.data.frame()

names(xgbAnalyse2Reshaped) = c("Index", "Variable", "Value")

ggplot(data=xgbAnalyse2Reshaped,
       aes(x=Index, y=Value, colour=Variable)) +
  geom_line() + facet_grid(Variable~.)


## Parameter Importance
importance = (xgb.importance(model = xgbTree))
str(importance)
par(mar=c(1,1,1,1))
xgb.plot.importance((importance))

fi = importance %>% as.data.frame() %>% arrange(desc(Gain))
ggplot(data=head(fi,10), aes(x=reorder(Feature,Gain), y=(Gain))) +
  geom_bar(stat="identity", fill="darkolivegreen3") + coord_flip() +
  theme(text = element_text(size=15)) +
  ylab("Feature Importance(Gain)")+
  xlab("")
  

#################################### End ####################################




