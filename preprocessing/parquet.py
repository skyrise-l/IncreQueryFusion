import pandas as pd
import os
import glob
from pathlib import Path
import time

def csv_to_parquet_batch(folder_path, output_folder=None, delete_original=False):
    """
    将文件夹中的所有CSV文件转换为Parquet格式
    
    参数:
    folder_path: 包含CSV文件的文件夹路径
    output_folder: 输出文件夹路径，如果为None则使用原文件夹
    delete_original: 是否在转换后删除原CSV文件
    """
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {folder_path} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件，开始转换...")
    
    success_count = 0
    error_files = []
    
    for csv_file in csv_files:
        try:
            start_time = time.time()
            
            # 生成输出文件名
            filename = Path(csv_file).stem
            parquet_file = os.path.join(output_folder, f"{filename}.parquet")
            
            print(f"正在转换: {filename}.csv -> {filename}.parquet")
            
            # 读取CSV文件（使用优化参数）
            df = pd.read_csv(
                csv_file,
                low_memory=False,
                engine='c'
            )
            
            # 转换为Parquet
            df.to_parquet(
                parquet_file,
                engine='pyarrow',  # 使用pyarrow引擎，性能更好
                compression='snappy',  # 快速压缩
                index=False
            )
            
            # 验证转换
            if os.path.exists(parquet_file):
                # 计算文件大小变化
                csv_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
                parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)  # MB
                compression_ratio = (csv_size - parquet_size) / csv_size * 100
                
                elapsed_time = time.time() - start_time
                
                print(f"✓ 转换成功: {filename}.parquet")
                print(f"  文件大小: {csv_size:.2f}MB → {parquet_size:.2f}MB (压缩 {compression_ratio:.1f}%)")
                print(f"  转换时间: {elapsed_time:.2f}秒")
                
                # 可选：删除原CSV文件
                if delete_original:
                    os.remove(csv_file)
                    print(f"  已删除原文件: {filename}.csv")
                
                success_count += 1
            else:
                raise Exception("Parquet文件创建失败")
                
        except Exception as e:
            print(f"✗ 转换失败: {filename}.csv - 错误: {str(e)}")
            error_files.append(csv_file)
    
    # 输出总结
    print("\n" + "="*50)
    print("转换总结:")
    print(f"成功: {success_count}/{len(csv_files)}")
    if error_files:
        print(f"失败的文件:")
        for file in error_files:
            print(f"  - {file}")
    
    return success_count, error_files

def compare_read_speed(csv_file, parquet_file, sample_rows=10000):
    """
    比较CSV和Parquet文件的读取速度
    """
    print(f"\n性能对比测试:")
    print(f"CSV文件: {csv_file}")
    print(f"Parquet文件: {parquet_file}")
    
    # 测试CSV读取速度
    start_time = time.time()
    df_csv = pd.read_csv(csv_file, nrows=sample_rows)
    csv_time = time.time() - start_time
    
    # 测试Parquet读取速度
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_file)
    parquet_time = time.time() - start_time
    
    # 确保数据一致
    assert df_csv.shape[0] == min(sample_rows, df_parquet.shape[0]), "数据行数不一致"
    
    speedup = csv_time / parquet_time
    
    print(f"CSV读取时间: {csv_time:.3f}秒")
    print(f"Parquet读取时间: {parquet_time:.3f}秒")
    print(f"速度提升: {speedup:.1f}倍")
    
    return speedup

# 使用示例
if __name__ == "__main__":
    # 示例1: 转换单个文件夹中的所有CSV文件
    folder_path = "/home/lwh/QueryFusion/data/dataset/movie/final_data_emd"  # 替换为你的文件夹路径
    
    # 执行转换
    success_count, errors = csv_to_parquet_batch(
        folder_path=folder_path,
        output_folder=None,  # 输出到原文件夹
        delete_original=False  # 谨慎使用，建议先设置为False测试
    )
    
