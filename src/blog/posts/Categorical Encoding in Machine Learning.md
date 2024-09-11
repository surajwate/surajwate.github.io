---
title: Categorical Encoding in Machine Learning
date: 2024-09-10
lastUpdated: 2024-09-10
tags: 
description: A brief summary of the post
categories: 
layout: layouts/post.html
permalink: /blog/categorical-encoding-in-machine-learning/
draft: true
---
## What is Categorical Encoding

Categorical encoding is the process of converting categorical (non-numeric) variable into numerical values so that they can be used by machine learning models. There are several techniques to achieve this, and the choice of technique depends on the type of categorical variable you're dealing with (nominal vs. ordinal) and the specific needs of machine learning model.


## Nominal vs Ordinal Variables




## Choosing the Right Encoding




## Conclusion

Categorical encoding is a crucial step in preparing data for machine learning models. By understanding the different encoding methods, their pros and cons, and when to use them, you can significantly improve your model's performance. Explore the individual articles linked above for more detailed information on each encoding method.

If you have any questions or want to dive deeper into specific techniques, feel free to explore the individual blog posts!


---



Encoding
│
├── Nominal Encoding (For Categorical Variables Without Order)
│   ├── ==One Hot Encoding==
│   ├── Dummy Encoding (One Hot Encoding with one category dropped)
│   ├── Mean Encoding (Target Encoding)
│   ├── ==Frequency Encoding==
│   ├── Binary Encoding
│   ├── Hash Encoding
│   └── Leave-One-Out Encoding (A variant of Target Encoding)
│
└── Ordinal Encoding (For Categorical Variables With Order)
    ├── ==Label Encoding==
    ├── Target Guided Ordinal Encoding
    ├── ==Ordinal Encoder== (Using an Ordered Map)
    ├── Count Encoding
    └── Helmert Encoding (Used to contrast levels of an ordinal variable)
