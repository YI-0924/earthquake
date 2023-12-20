import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier     # randomforest = bagging + decision tree
from sklearn.metrics import f1_score, precision_score, recall_score

# read_csv
df = pd.read_csv("train_values.csv")
'''
print(df.isnull().sum())    # 看有沒有缺失值
print(df[''].describe())    # 看各個Feature詳細的資訊(資料分布)
for i in df:
    df[i].plot.box()
    plt.show()              # 看outliers
'''

lbl = pd.read_csv("train_labels.csv")
'''
lbl.damage_grade.value_counts().plot.bar(title='Number of building with each damage grade')
plt.show()
selected_features = ['age', 'area_percentage', 'height_percentage', 'count_families']
train_values_subset = df[selected_features]
sns.pairplot(train_values_subset.join(lbl), 
             hue='damage_grade')
plt.show()
'''

test_value = pd.read_csv("test_values.csv")
'''
print(test_value.isnull().sum())    # 看有沒有缺失值
'''

# change categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['land_surface_condition', 'foundation_type',
'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
'legal_ownership_status'])
test_value = pd.get_dummies(test_value, columns=['land_surface_condition', 'foundation_type',
'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
'legal_ownership_status'])

# combine 'has_secondary_use' & 'has_secondary_use_APPLICATION' as 1 column
# if 'has_secondary_use_APPLICATION' is 1, change 'has_secondary_use' into 2 to 11, depends on its application type
# so only 0, 2~11 will exist, 1 will NOT exist
# cnt=0 # test for warning ignorance
def change_has_sec_use(data):
    cpy = data['has_secondary_use'].copy()    # 先用copy才不會報warning
    for i in range(data.shape[0]):
        if data['has_secondary_use'][i] == 1:
            # cnt+=1
            if data['has_secondary_use_agriculture'][i] == 1:
                cpy[i] = 2
            elif data['has_secondary_use_hotel'][i] == 1:
                cpy[i] = 3
            elif data['has_secondary_use_rental'][i] == 1:
                cpy[i] = 4
            elif data['has_secondary_use_institution'][i] == 1:
                cpy[i] = 5
            elif data['has_secondary_use_school'][i] == 1:
                cpy[i] = 6
            elif data['has_secondary_use_industry'][i] == 1:
                cpy[i] = 7
            elif data['has_secondary_use_health_post'][i] == 1:
                cpy[i] = 8
            elif data['has_secondary_use_gov_office'][i] == 1:
                cpy[i] = 9
            elif data['has_secondary_use_use_police'][i] == 1:
                cpy[i] = 10
            elif data['has_secondary_use_other'][i] == 1:
                cpy[i] = 11
    data['has_secondary_use'] = cpy
    # drop 'has_secondary_use_APPLICATION' columns since their info has been saved into 'has_secondary_use'
    data = data.drop(columns=['has_secondary_use_agriculture','has_secondary_use_hotel',
    'has_secondary_use_rental', 'has_secondary_use_institution','has_secondary_use_school',
    'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
    'has_secondary_use_use_police', 'has_secondary_use_other'])

change_has_sec_use(df)
change_has_sec_use(test_value)

# drop n/a values
df = df.dropna()

# test if warning can be ignored
"""
count=[]
for i in range(2,12):
    count.append (df['has_secondary_use'].value_counts()[i])
sum=0
for i in range(len(count)):
    sum+=count[i]
print(cnt)
print(sum)
"""

'''
info = []
for i in range(11):
    info.append([0,0,0])

for i in range(df.shape[0]):
    if df['has_secondary_use'][i]==0:
        if lbl['damage_grade'][i] == 1:
            info[0][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[0][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[0][2]+=1
    elif df['has_secondary_use'][i]==2:
        if lbl['damage_grade'][i] == 1:
            info[1][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[1][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[1][2]+=1
    elif df['has_secondary_use'][i]==3:
        if lbl['damage_grade'][i] == 1:
            info[2][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[2][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[2][2]+=1

    elif df['has_secondary_use'][i]==4:
        if lbl['damage_grade'][i] == 1:
            info[3][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[3][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[3][2]+=1

    elif df['has_secondary_use'][i]==5:
        if lbl['damage_grade'][i] == 1:
            info[4][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[4][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[4][2]+=1
    
    elif df['has_secondary_use'][i]==6:
        if lbl['damage_grade'][i] == 1:
            info[5][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[5][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[5][2]+=1

    elif df['has_secondary_use'][i]==7:
        if lbl['damage_grade'][i] == 1:
            info[6][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[6][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[6][2]+=1

    elif df['has_secondary_use'][i]==8:
        if lbl['damage_grade'][i] == 1:
            info[7][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[7][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[7][2]+=1

    elif df['has_secondary_use'][i]==9:
        if lbl['damage_grade'][i] == 1:
            info[8][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[8][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[8][2]+=1

    elif df['has_secondary_use'][i]==10:
        if lbl['damage_grade'][i] == 1:
            info[9][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[9][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[9][2]+=1
    elif df['has_secondary_use'][i]==11:
        if lbl['damage_grade'][i] == 1:
            info[10][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[10][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[10][2]+=1

fig,ax= plt.subplots(4, 3) 

for i in range(4):
    for j in range(3):
        if(3*i+j < 11):
            ax[i,j].plot(info[3*i+j])
plt.show()
'''

# drop building_id(用不到)
building_id = test_value['building_id']     # submission.csv用
df = df.drop(['building_id'], axis=1)
lbl = lbl.drop(['building_id'], axis=1)
test_value = test_value.drop(['building_id'], axis=1)
lbl = lbl.values.ravel()    # 避免dataConversionWarning

# 在少數樣本附近人工創造一些樣本(根據鄰近點，不限少數樣本)，避免數據不平衡
df, lbl = BorderlineSMOTE(random_state=42, kind='borderline-2').fit_resample(df, lbl)

# 找Randomforest參數
'''
score_pic = []
for i in range(0, 200, 10):
    rfc = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=i+1, random_state=42))
    score = cross_val_score(rfc, df, lbl, cv=5).mean()   # 用交叉驗證計算得分
    score_pic.append(score)
score_max = max(score_pic)
print("highest score: ",score_max, "n_estimators: ", score_pic.index(score_max)*10+1)
x = np.arange(1, 201, 10)
plt.subplot(111)
plt.plot(x, score_pic, 'r-')
plt.show()
'''

gs = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_estimators=191, min_samples_leaf=1))   # 串多個模型, standardscaler標準化(高斯分布)
# param_grid = {'randomforestclassifier__min_samples_leaf':[1, 5]}  # 需要最佳參數的取值
# gs = GridSearchCV(gs, param_grid, cv=5)   # 網格搜索(窮舉), cv交叉驗證參數(分出訓練s跟測試集訓練模型)
gs.fit(df, lbl)   # training
pred = gs.predict(df)
print(f1_score(lbl, pred, average='micro'))
# print(gs.best_params_, gs.best_score_)

# create submission.csv
predictions = gs.predict(test_value)
submission = pd.DataFrame(data=predictions, columns=['damage_grade'])
submission.insert(0, 'building_id', building_id)
submission.to_csv("submission.csv", index=False)