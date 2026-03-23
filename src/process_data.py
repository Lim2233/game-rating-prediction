import pandas as pd

def processData():
    
    # 读取原始数据
    df = pd.read_csv('data/raw/games_raw.csv')
    
    #去除多余的列
    cols_to_drop = ['img', 'last_update', 'title','na_sales','jp_sales','pal_sales','other_sales']
    df.drop(columns=cols_to_drop, inplace=True)


    #去除空行
    for i in df.columns:
        df.dropna(subset=[i],inplace = True)
    
    
    #处理日期
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y')
    ddl = pd.Timestamp('2024-01-01')
    df['days_since_release'] = (ddl - df['release_date']).dt.days
    df.drop(columns='release_date',inplace = True)
    
    #保存数据
    save_road='data/processed/data_processed.csv'
    df.to_csv(save_road,index=False)
    
    print('[processData]:Already!')


if __name__ == "__main__":
    processData()