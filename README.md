# Starbucks_Capstone_notebook

**Table Of Contents**

1. [Installation](#installation)
2. [Project Overview](#project)
3. [File Descripton](#file)
4. [Results](#results)
5. [Acknowledge](#ackowledge)

## Installation<a id="installation"></a>

The project is supported by Python 3.x (Hint: directly install the Anaconda distribution), libraries and packages:

- Numpy
- Pandas
- Seaborn
- Matplotlib
- Sci-kit tool

## Project Overview<a id="project"></a>

The data set contains three files, which are simulated on the Starbucks rewards mobile app, which focuses on the customers' transactions behavior. Our topic targets are that how the effect of the reward offer is and is there a relationship between the reward offer and the users' demographic information. From the company side, it is interesting that whether the customers are more willing to pay more money in the Starbucks after the promotion.

When we analyze the users' willing, we use the amount as the target label and the users' demographic information and the promotion type as explanatory variables. Of course, the target label is continuous value, we consider that the Supervised learning is a right way to build model.

## File Description<a id="file"></a>

* [Starbucks_Capstone_notebook.ipynb](./Starbucks_Capstone_notebook.ipynb)

  The notebook is used to run code, which contains data wrangling, data exploration, and model build

* [tool](./tool)

  The directory contains all functional script

* [Starbucks_Capstone_notebook_report.md](./Starbucks_Capstone_notebook_report.md)

  The final report file

```
├── README.md
├── Starbucks_Capstone_notebook.ipynb
├── data
│   ├── portfolio.json
│   ├── profile.json
│   └── transcript.json
└── tool
    ├── addiction.py
    ├── model.py
    └── preprocess.py
```



## Results <a id="results"></a>

1. **Effect Of Different Promotion**
   The total number of the users is **17,000**. But after the promotion done, there are **4,332** members actively going to the Starbucks. The ratio is **25.48%**.

   - The amount of promotion strategy is almost the same.
     1. The largest one is the discount has **10,101**
     2. The smallest one is the **BOGO** has **8,568**
     3. The informational is the middle one, has **9,548**
   - But there is a significant  growth rate
     1. In totally, there is a positive effect at each strategy
     2. The **informational** make a high improve to attract the customers


3. **Model Result**

   We build three models: the linear regression is a basic model which we got test data is: 1.1829, R^2 score of the test data is: 0.4732 and train data is: 0.4684.


## Acknowledge <a id="acknowlege"></a>

Besides, you must know that the project is under the [Udacity Data Science Capstone Project](https://classroom.udacity.com/nanodegrees/nd025/parts/84260e1f-2926-4127-895f-cc4432b05059/modules/80c955ce-72f2-403a-9bf5-cc58636dab9d/lessons/d6285247-6bc0-4783-b118-6f41981b9469/concepts/480e9dc2-4726-4582-81d7-3b8e6a863450), and the dataset is provied by the Starbucks.
