# Used-Car Listing Price vs. Market Positioning Simulator

**Author:** Hitansh Kumar Kanther (PGDM — Big Data Analytics, Goa Institute of Management)  
**Course:** Advanced Machine Learning for Business Transformation (AMLBT)

## Overview
End-to-end machine learning decision-support system for Indian used-car pricing ($N = 8,631$ listings). Combines domain text feature engineering (`AdditionInfo` regex parsing) with Gradient Boosting Quantile Regression ($q_{20}, q_{50}, q_{80}$) to benchmark asking prices across **Fast Sale ($q_{20}$)**, **Balanced ($q_{50}$)**, and **Maximum Price ($q_{80}$)** seller strategies.