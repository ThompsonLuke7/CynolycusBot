import pandas as pd
import os

def main():
    global_file_path = "C:/Users/luket/CynolycusBot/Data/processed/"
    X_df = pd.read_parquet(os.path.join(global_file_path,"X_spy_daily.parquet"))
    labels_df = pd.read_parquet(os.path.join(global_file_path,"labels_spy_daily.parquet"))

    df = pd.concat([X_df, labels_df], axis=1)
    df = df.reset_index(drop=True)
    
    print(df[['close','atr_swing_label']].head(20))
    print(df[['close','atr_swing_label']].tail(20))
    
    print(df.isna().sum().sort_values())
    print(df.index[:10])
    print(df.index[-10:])
    
    for col in df.columns:
        if df[col].equals(df['atr_swing_label']):
            print("LEAK:", col)
            
    corr = df.corr()
    print(corr['atr_swing_label'].sort_values(ascending=False).head(20))





if __name__ == "__main__":
    main()
