# Extension: Applying to UCI Dataset

The original Ahmed (2024) dataset (Wollo University) is replaced with:
> **UCI Student Performance Dataset (Kaggle)**  
> https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

### Attribute Mapping
| Original Feature | UCI Equivalent | Description |
|------------------|----------------|--------------|
| gender | sex | student's gender |
| region | address | urban or rural |
| entrance_result | G1 / G2 | prior exam results |
| attempts | failures | number of past course failures |
| studied_credits | studytime | weekly study time |
| disability | — | not available |
| final_result | G3 | final grade (converted to Pass/Fail) |

The same ML pipeline (RandomForest + SVM/DT/KNN/NB) was reused unchanged.
