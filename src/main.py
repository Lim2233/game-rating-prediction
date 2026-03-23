import process_data as pdata
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor, Pool


pdata.processData()

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
# 获取所有字符串类型的列名
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()


model = CatBoostRegressor(
    iterations=1500,           # 树的数量（迭代次数）
    learning_rate=0.05,       # 学习率
    depth=8,                  # 树的深度
    loss_function='RMSE',     # 回归任务常用 RMSE
    cat_features=categorical_features,  # 传入类别特征
    verbose=100,              # 每100轮输出一次日志
    early_stopping_rounds=100  # 早停，防止过拟合
)

# 训练模型（使用 Pool 不是必须的，但可以更方便地传递参数）
model.fit(X_train, y_train, eval_set=(X_test, y_test))



#评估模型性能
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")