# Data Driven System To Predict Grades And Dropout
A thesis submitted to the School of Computer Science and Engineering in partial fulfillment of the requirements for the degree of Bachelor of Information Technology.
# Objective
- In this study, a dataset collected from the University will be used to build a predictive model using various machine learning algorithms to analyze and evaluate student’s data. They are built based on a general assumption that the pre-requisite or outline course’s knowledge can affect the achievement of future courses, so that these can be used to predict the academic grade of students. we proposed another algorithm RNN for this course-specific model which not only applies for the result data, but also focuses on the knowledge’s evolution cross sequence of semesters. The unbalanced dataset issue is one of the key elements influencing classifier performance. In the realm of EDM, it is a significant challenge that might result in inaccurate outcomes and poor performance.  
- In this case, we choose the ‘Data Structures and Algorithms’ (DSA) as the target course to build course- specific models. As mentioned above, After that, 11 prerequisite, outline or related courses are used as the features for these models. Moreover, There are three class labels for the grade level prediction, including Good, Average and Fail and There are two class labels for the dropout rate, including Risk and Not-risk.
  ![image](https://github.com/AnhTuan160301/DataDrivenSystemToPredictGradesAndDropout/assets/74240275/bd2fc70e-5c22-4761-ad5b-7179d17adc19)
# Results
- **Dropout Prediction:** For the dropout or leaving a course early, a traditional machine learning technique, namely Support Vector Machine will be implemented based on the tabular data of 11 previous pre-requisite and outline courses. The overall results, this predictive model can reach **81%** of accuracy and **90.06%** and **80.09%** for the Precision and Recall.
- **Grade Prediction:** Through the 175 epochs, our learning model’s accuracy is about **84.12%** and the overall loss is **0.4858**.
# Requirements
To run the code, ensure you have:
* Python installed
* JuPyter Notebooks or Python IDE
* Necessary libraries: ```sklearn```, ```NumPy```, ```pandas```, ```matplotlib```, ```seaborn``` and ```Streamlit```
# Steps For Implementing Grade Dropout System
1. Install Project.
2. Navigate to Project folder
3. Install some require libraries
- ```pip install Name Of Library```
4. Run the ```app.py``` python file
- ``` streamlit run "Directory Path Of app.py" ```
5. Go to your Browser and enter webserver address
- ```localhost:8501```
