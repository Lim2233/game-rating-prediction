import process_data as pdata
import split_data as sdata
import train
from sklearn.model_selection import train_test_split

#from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor


pdata.processData()
sdata.splitData()
train.train()

# #评估模型性能
# y_pred = model.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)

# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")