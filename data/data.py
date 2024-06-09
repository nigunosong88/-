import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
from scipy.stats import norm, skew 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def data_csv():
    # 設定 matplotlib 的字體
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 'Microsoft YaHei' 作為例子，可更換成其他字體
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

    df = pd.read_csv('E_lvr_land_A.csv')
    df = df.drop(['sign','address','non-metropolis','non-metropolis2',
                  'main use','the unit price NTD','the note','serial number',
                  'ID','main building materials'], axis=1)
    train_data = df.dropna(subset =  ['theUseZoningorCompiles'])
    train_data = train_data.dropna(subset =  ['ShiftingfloorNumber'])
    train_data = train_data.dropna(subset =  ['floor'])
    train_data = train_data.dropna(subset =  ['ConstructionYear'])
    ##########################
    plt.pie(train_data['elevator'].value_counts(), radius=1.5, labels=train_data['elevator'].unique(),autopct='%.1f%%')
    plt.title("有無電梯")
    plt.savefig('figures/電梯.jpeg', bbox_inches='tight')
    value_counts = train_data['District'].value_counts()

    threshold = 0.03 * len(train_data)
    other_count = sum(value_counts[value_counts < threshold])

    value_counts_filtered = value_counts[value_counts >= threshold]
    value_counts_filtered['其他'] = other_count

    plt.pie(value_counts_filtered, radius=1.5, labels=value_counts_filtered.index, autopct=lambda p: '{:.1f}%'.format(p) if p >= 3 else '')
    plt.title("區")
    plt.savefig('figures/區域.jpeg', bbox_inches='tight')
    live=train_data.District.unique()
    x = np.arange(len(live))
    # 使用 Matplotlib 的 'rcParams' 配置參數來設置字體支持中文顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 'Microsoft YaHei' 是一種常用的中文字體
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
    plt.figure(figsize=(20, 6))  # 調整圖表大小
    plt.bar(train_data.District.unique(),
        train_data.District.value_counts(), 
        align='center', 
        color=['lightsteelblue', 
               'cornflowerblue'])   
    plt.xticks(x ,live,rotation=45) #rotation旋轉標籤
    plt.savefig('figures/區域2.jpeg', bbox_inches='tight')
    filtered_df = train_data[train_data['PriceNTD'] < 1e8]
    plt.figure(figsize=(16, 8))
    plt.scatter(x=filtered_df['main building area'], y=filtered_df['PriceNTD'])
    plt.ylim=(0,800000)  # y坐标轴范围
    plt.xlabel('Grlivarea ')  # x轴名称
    plt.ylabel('SalePrice ')  # y轴名称
    plt.title(' Grlivarea and SalePrice') #标题
    plt.savefig('figures/價格.jpeg', bbox_inches='tight')
    ##########################
    train_data['BerthCategory']= train_data['BerthCategory'].fillna("無")

    labelencoder = LabelEncoder()
    train_data['District'] = labelencoder.fit_transform(train_data['District'].values)
    train_data['theUseZoningorCompiles'] = labelencoder.fit_transform(train_data['theUseZoningorCompiles'].values)
    train_data['BuildingState'] = labelencoder.fit_transform(train_data['BuildingState'].values)
    train_data['Compartment'] = labelencoder.fit_transform(train_data['Compartment'].values)
    train_data['ManagementOrganization'] = labelencoder.fit_transform(train_data['ManagementOrganization'].values)
    train_data['elevator'] = labelencoder.fit_transform(train_data['elevator'].values)
    train_data['BerthCategory'] = labelencoder.fit_transform(train_data['BerthCategory'].values)

    # 正則表達式用來匹配 "土地X建物Y車位Z" 格式，其中 X, Y, Z 為數字
    train_data[['Land', 'Building', 'Berth']] = train_data['TransactionPenNumber'].str.extract('土地(\d+)建物(\d+)車位(\d+)')
    # 轉換數據類型，因為提取出來的數據預設是字符串
    train_data[['Land', 'Building', 'Berth']] = train_data[['Land', 'Building', 'Berth']].astype(int)
    train_data = train_data.drop(['TransactionPenNumber'], axis=1)

    train_data['ConstructionYear']=train_data['ConstructionYear'].astype(int)
    train_data['Year'] = train_data['ConstructionYear'].apply(lambda x: str(x)[:3] if len(str(x)) == 7 else str(x)[:2])
    train_data['Year']=113-train_data['Year'].astype(int)

    def chinese_to_arabic(cn):
        digits = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        if cn == '全':
            return None  
        elif cn.startswith('十'):
            return 10 + digits.get(cn[1:], 0)
        elif cn.endswith('十'):
            return digits.get(cn[:-1], 1) * 10
        else:
            return digits.get(cn, None)

    train_data['ShiftingfloorNumber'] = train_data['ShiftingfloorNumber'].apply(lambda x: chinese_to_arabic(x.replace('層', '')))
    train_data['floor'] = train_data['floor'].apply(lambda x: chinese_to_arabic(x.replace('層', '')))

    train_data['ShiftingfloorNumber'].fillna(0, inplace=True)
    train_data['floor'].fillna(0, inplace=True)

    # 正态化Y
    train_data['PriceNTD']= np.log1p(train_data['PriceNTD'])

    train_data = train_data.drop(['Land','theUseZoningorCompiles','TransactionYearMonthDay',
                                  'ManagementOrganization','elevator','BerthCategory','Year'], axis=1)
    
    X = train_data[['District', 'LandShiftingTotalArea','ShiftingfloorNumber', 'floor',
       'BuildingState', 'ConstructionYear', 'BuildingShiftingTotalArea',
       'Room', 'Hall', 'Bathroom', 'Compartment','BerthShiftingTotalArea',
       'BerthTotalPriceNTD', 'main building area', 'auxiliary building area',
       'balcony area','Building', 'Berth']]
    y = train_data['PriceNTD']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X,y