import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
def train():
    
    df=pd.read_csv('data/processed/data_processed.csv')

    X_train=pd.read_csv('data/processed/sliced/X_train.csv')
    X_test=pd.read_csv('data/processed/sliced/X_test.csv')
    y_train=pd.read_csv('data/processed/sliced/y_train.csv')
    y_test=pd.read_csv('data/processed/sliced/y_test.csv')

    print('[main]:Read Successfullily')

    # 定义特征列和目标列
    features = ['console', 'genre', 'publisher', 'developer', 'total_sales', 'days_since_release']
    target = 'critic_score'

    X = df[features]
    y = df[target]

    # 获取所有字符串类型的列名
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42
    )

    model = CatBoostRegressor(
        iterations=500,           # 树的数量（迭代次数）
        learning_rate=0.05,       # 学习率
        depth=6,                  # 树的深度
        loss_function='RMSE',     # 回归任务常用 RMSE
        cat_features=categorical_features,  # 传入类别特征
        verbose=25,              # 每25轮输出一次日志
        early_stopping_rounds=100  # 早停，防止过拟合
    )

    # 训练模型
    model.fit(X_train, y_train, eval_set=(X_test, y_test))


    #保存模型
    model.save_model('./models/model.cbm')
    print('[train]:Already!')
    
if __name__ =='__main__':
    train()
    
    