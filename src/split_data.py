import pandas as pd

from sklearn.model_selection import train_test_split

def splitData():
    
    df=pd.read_csv('data/processed/data_processed.csv')


    # 定义特征列和目标列
    features = ['console', 'genre', 'publisher', 'developer', 'total_sales', 'days_since_release']
    target = 'critic_score'

    X = df[features]
    y = df[target]


    #分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42
    )
    
    #保存为 CSV 文件
    
    X_train.to_csv('./data/processed/sliced/X_train.csv')
    X_test.to_csv('./data/processed/sliced/X_test.csv')
    y_train.to_csv('./data/processed/sliced/y_train.csv')
    y_test.to_csv('./data/processed/sliced/y_test.csv')
    
    print('[splitData]:Already!')
    
if __name__ =='__main__':
    splitData()
    
    