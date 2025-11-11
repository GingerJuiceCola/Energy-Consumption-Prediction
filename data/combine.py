import pandas as pd
import os
import numpy as np

def process_energy_data(file_path):
    #稳健处理能耗数据文件，按小时计算平均值
    print(f"处理文件: {file_path}")
    
    chunks = []
    chunk_size = 50000
    
    try:
        chunk_reader = pd.read_csv(
            file_path,
            usecols=lambda col: col not in ['fecha_esp32', 'MAC', 'weekday'],
            quotechar='"',
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='skip'
        )
        
        for i, chunk in enumerate(chunk_reader):
            # 转换时间格式
            chunk['fecha_servidor'] = pd.to_datetime(chunk['fecha_servidor'], errors='coerce')
            chunk = chunk.dropna(subset=['fecha_servidor'])
            
            if chunk.empty:
                continue
                
            # 确保所有列都是数值类型
            for col in chunk.columns:
                if col != 'fecha_servidor':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # 设置时间索引并按小时重采样
            chunk.set_index('fecha_servidor', inplace=True)
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk_numeric = chunk[numeric_cols]
            
            chunk_hourly = chunk_numeric.resample('h').mean()
            chunks.append(chunk_hourly)
            
    except Exception as e:
        print(f"处理过程中遇到错误: {e}")
        print("继续使用已成功读取的数据...")
    
    if not chunks:
        print("没有成功读取任何数据块")
        return None
    
    # 合并所有块
    combined = pd.concat(chunks, axis=0)
    final_data = combined.resample('h').mean().reset_index()
    final_data.rename(columns={'fecha_servidor': 'datetime'}, inplace=True)
    
    # 保留5位小数
    for col in final_data.columns:
        if col != 'datetime' and pd.api.types.is_numeric_dtype(final_data[col]):
            final_data[col] = final_data[col].round(5)
    
    print(f"成功处理了 {len(chunks)} 个数据块，最终形状: {final_data.shape}")
    return final_data

def main():
    # 文件路径※※※
    energy1_path = "1mayo - agosto 2021.csv"
    energy2_path = "2agosto -dic 2021.csv"
    weather_path = "weather2021_05to12.csv"
    final_output_path = "finaldata1.csv"
    
    # 处理两个能源数据集
    energy1 = process_energy_data(energy1_path)
    energy2 = process_energy_data(energy2_path)
    
    if energy1 is None or energy2 is None:
        print("能源数据处理失败")
        return
    
    # 合并能源数据
    energy_combined = pd.concat([energy1, energy2], ignore_index=True)
    energy_combined.sort_values('datetime', inplace=True)
    
    # 读取天气数据
    print(f"读取天气数据: {weather_path}")
    try:
        weather = pd.read_csv(weather_path)
        weather['datetime'] = pd.to_datetime(weather['datetime'])     
        print(f"天气数据形状: {weather.shape}")
    except Exception as e:
        print(f"读取天气数据失败: {e}")
        return
    
    # 合并能源和天气数据
    print("合并能源和天气数据...")
    final_data = pd.merge(energy_combined, weather, on='datetime', how='inner')
    final_data.to_csv(final_output_path, index=False)
    
    # 显示时间范围和数据完整性
    if not final_data.empty:
        print(f"数据时间范围: {final_data['datetime'].min()} 到 {final_data['datetime'].max()}")
        
        energy_hours = len(energy_combined)
        weather_hours = len(weather)
        final_hours = len(final_data)
        
        print(f"数据完整性: 能源{energy_hours}小时, 天气{weather_hours}小时, 合并{final_hours}小时")
        print(f"数据覆盖率: {final_hours/min(energy_hours, weather_hours)*100:.1f}%")

if __name__ == "__main__":
    main()
