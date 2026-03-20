import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁用图形显示窗口，只保存文件
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from datetime import datetime

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义咖啡饮品列表
COFFEE_DRINKS = ["美式咖啡", "拿铁咖啡", "卡布奇诺", "摩卡咖啡", "焦糖玛奇朵", "抹茶拿铁"]

def load_and_validate_data(file_path):
    """加载并验证数据"""
    df = pd.read_excel(file_path)
    
    # 1. 数据校验：检查消费总额是否等于消费杯数×单价
    df['计算消费总额'] = df['消费杯数'] * df['单价(元)']
    inconsistent = df[df['消费总额(元)'] != df['计算消费总额']]
    
    if len(inconsistent) > 0:
        print(f"发现 {len(inconsistent)} 条数据计算不一致，已自动修正")
        df['消费总额(元)'] = df['计算消费总额']
    else:
        print("数据校验通过：所有记录消费总额计算一致")
    
    # 移除辅助列
    df = df.drop('计算消费总额', axis=1)
    
    # 添加月份列用于后续聚合
    df['消费日期'] = pd.to_datetime(df['消费日期'])
    df['月份'] = df['消费日期'].dt.to_period('M')
    df['月份名称'] = df['消费日期'].dt.strftime('%Y-%m')
    
    return df

def monthly_aggregation(df):
    """按月度聚合汇总"""
    # 维度1：按消费项目聚合
    monthly_by_item = df.groupby(['月份名称', '消费项目'])['消费总额(元)'].sum().unstack().fillna(0)
    # 维度2：按消费区域聚合
    monthly_by_area = df.groupby(['月份名称', '消费区域'])['消费总额(元)'].sum().unstack().fillna(0)
    # 月度总消费
    monthly_total = df.groupby('月份名称')['消费总额(元)'].sum()
    
    return monthly_by_item, monthly_by_area, monthly_total

def consumption_ratio_analysis(df):
    """消费占比分析"""
    total_consumption = df['消费总额(元)'].sum()
    
    # 各消费区域占比
    area_ratio = df.groupby('消费区域')['消费总额(元)'].sum() / total_consumption * 100
    area_ratio = area_ratio.sort_values(ascending=False)
    
    # 各消费项目占比
    item_ratio = df.groupby('消费项目')['消费总额(元)'].sum() / total_consumption * 100
    item_ratio = item_ratio.sort_values(ascending=False)
    
    return area_ratio, item_ratio, total_consumption

def barista_performance(df):
    """咖啡师业绩排名（仅统计咖啡饮品）"""
    coffee_df = df[df['消费项目'].isin(COFFEE_DRINKS)]
    total_coffee = coffee_df['消费总额(元)'].sum()
    
    barista_stats = coffee_df.groupby('咖啡师ID')['消费总额(元)'].sum().sort_values(ascending=False)
    barista_contribution = barista_stats / total_coffee * 100
    
    barista_result = pd.DataFrame({
        '消费总额(元)': barista_stats,
        '贡献度(%)': barista_contribution.round(2)
    }).head(3)
    
    return barista_result, total_coffee

def age_preference_analysis(df):
    """年龄区间消费偏好分析"""
    # 各年龄区间消费项目分布
    age_item_dist = df.groupby(['年龄区间', '消费项目'])['消费总额(元)'].sum().unstack().fillna(0)
    age_item_ratio = age_item_dist.div(age_item_dist.sum(axis=1), axis=0) * 100
    
    # 各年龄区间人均消费额
    age_total = df.groupby('年龄区间')['消费总额(元)'].sum()
    age_members = df.groupby('年龄区间')['会员ID'].nunique()
    age_avg_consumption = (age_total / age_members).round(2)
    
    return age_item_ratio, age_avg_consumption, age_members

def plot_monthly_trend(monthly_by_item, monthly_by_area, monthly_total):
    """绘制月度消费总额趋势图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # 1. 月度总消费趋势
    ax0 = axes[0]
    monthly_total.plot(kind='line', marker='o', ax=ax0, color='#2c3e50', linewidth=2, markersize=8)
    ax0.set_title('月度消费总额趋势', fontsize=16, pad=20)
    ax0.set_ylabel('消费总额(元)', fontsize=12)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='x', rotation=45)
    
    # 2. 各消费项目月度趋势
    ax1 = axes[1]
    monthly_by_item.plot(kind='line', marker='o', ax=ax1, linewidth=2, markersize=6)
    ax1.set_title('各消费项目月度趋势', fontsize=16, pad=20)
    ax1.set_ylabel('消费总额(元)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 3. 各消费区域月度趋势
    ax2 = axes[2]
    monthly_by_area.plot(kind='line', marker='o', ax=ax2, linewidth=2, markersize=6)
    ax2.set_title('各消费区域月度趋势', fontsize=16, pad=20)
    ax2.set_ylabel('消费总额(元)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('月度消费趋势分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_consumption_ratio(area_ratio, item_ratio):
    """绘制消费占比图"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 消费区域占比饼图
    ax0 = axes[0]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    wedges, texts, autotexts = ax0.pie(area_ratio.values, labels=area_ratio.index, 
                                       autopct='%1.1f%%', colors=colors[:len(area_ratio)])
    ax0.set_title('各消费区域消费额占比', fontsize=16, pad=20)
    
    # 消费项目占比条形图
    ax1 = axes[1]
    item_ratio.plot(kind='barh', ax=ax1, color='#3498db')
    ax1.set_title('各消费项目消费额占比(%)', fontsize=16, pad=20)
    ax1.set_xlabel('占比(%)', fontsize=12)
    
    # 在条形上显示数值
    for i, v in enumerate(item_ratio.values):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('消费占比分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_age_preference(age_item_ratio, age_avg_consumption):
    """绘制年龄区间消费偏好图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 消费项目分布热力图
    ax0 = axes[0]
    sns.heatmap(age_item_ratio, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax0)
    ax0.set_title('各年龄区间消费项目分布占比(%)', fontsize=16, pad=20)
    ax0.tick_params(axis='x', rotation=45)
    
    # 人均消费额条形图
    ax1 = axes[1]
    age_avg_consumption.plot(kind='bar', ax=ax1, color='#2ecc71')
    ax1.set_title('各年龄区间人均消费额(元)', fontsize=16, pad=20)
    ax1.set_ylabel('人均消费额(元)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 在条形上显示数值
    for i, v in enumerate(age_avg_consumption.values):
        ax1.text(i, v + 2, f'{v:.0f}元', ha='center')
    
    plt.tight_layout()
    plt.savefig('年龄消费偏好分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_barista_performance(barista_result):
    """绘制咖啡师业绩图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(barista_result.index, barista_result['消费总额(元)'], color=['#e74c3c', '#f39c12', '#27ae60'])
    ax.set_title('咖啡师业绩Top3', fontsize=16, pad=20)
    ax.set_ylabel('咖啡饮品消费总额(元)', fontsize=12)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(barista_result.iterrows()):
        ax.text(i, row['消费总额(元)'] + 50, 
                f'{row["消费总额(元)"]:.0f}元\n({row["贡献度(%)"]}%)',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('咖啡师业绩排名.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 1. 加载并验证数据
    print("=" * 60)
    print("第一步：数据校验与修正")
    print("=" * 60)
    df = load_and_validate_data(r'C:\Users\32767\Desktop\3.20\main\咖啡店会员消费数据.xlsx')
    print(f"数据记录数：{len(df)} 条")
    
    # 2. 月度聚合汇总
    print("\n" + "=" * 60)
    print("第二步：月度聚合汇总")
    print("=" * 60)
    monthly_by_item, monthly_by_area, monthly_total = monthly_aggregation(df)
    
    print("\n月度消费总额汇总：")
    print(monthly_total.round(2).to_string())
    
    print("\n按消费项目月度汇总（部分）：")
    print(monthly_by_item.round(2).head())
    
    print("\n按消费区域月度汇总：")
    print(monthly_by_area.round(2))
    
    # 3. 统计分析 - 消费占比
    print("\n" + "=" * 60)
    print("第三步：统计分析 - 消费占比分析")
    print("=" * 60)
    area_ratio, item_ratio, total_consumption = consumption_ratio_analysis(df)
    
    print(f"\n总消费额：{total_consumption:.2f} 元")
    print("\n各消费区域占比：")
    print(area_ratio.round(2).to_string() + " %")
    
    print("\n各消费项目占比：")
    print(item_ratio.round(2).to_string() + " %")
    
    # 4. 统计分析 - 咖啡师业绩排名
    print("\n" + "=" * 60)
    print("第四步：统计分析 - 咖啡师业绩排名")
    print("=" * 60)
    barista_result, total_coffee = barista_performance(df)
    
    print(f"\n咖啡饮品总消费额：{total_coffee:.2f} 元")
    print("\n咖啡师业绩Top3：")
    print(barista_result.to_string())
    
    # 5. 统计分析 - 年龄区间消费偏好
    print("\n" + "=" * 60)
    print("第五步：统计分析 - 年龄区间消费偏好分析")
    print("=" * 60)
    age_item_ratio, age_avg_consumption, age_members = age_preference_analysis(df)
    
    print("\n各年龄区间会员数：")
    print(age_members.to_string())
    
    print("\n各年龄区间人均消费额：")
    print(age_avg_consumption.to_string() + " 元")
    
    print("\n各年龄区间消费项目分布占比（%）：")
    print(age_item_ratio.round(2).to_string())
    
    # 6. 可视化
    print("\n" + "=" * 60)
    print("第六步：生成可视化图表")
    print("=" * 60)
    
    plot_monthly_trend(monthly_by_item, monthly_by_area, monthly_total)
    plot_consumption_ratio(area_ratio, item_ratio)
    plot_age_preference(age_item_ratio, age_avg_consumption)
    plot_barista_performance(barista_result)
    
    print("\n✅ 可视化图表已生成：")
    print("  - 月度消费趋势分析.png")
    print("  - 消费占比分析.png")
    print("  - 年龄消费偏好分析.png")
    print("  - 咖啡师业绩排名.png")
    
    # 7. 导出分析结果到Excel
    print("\n" + "=" * 60)
    print("第七步：导出分析结果")
    print("=" * 60)
    
    with pd.ExcelWriter('咖啡店数据分析结果.xlsx') as writer:
        monthly_total.to_excel(writer, sheet_name='月度总消费')
        monthly_by_item.to_excel(writer, sheet_name='按项目月度汇总')
        monthly_by_area.to_excel(writer, sheet_name='按区域月度汇总')
        pd.DataFrame(area_ratio, columns=['占比(%)']).to_excel(writer, sheet_name='消费区域占比')
        pd.DataFrame(item_ratio, columns=['占比(%)']).to_excel(writer, sheet_name='消费项目占比')
        barista_result.to_excel(writer, sheet_name='咖啡师业绩Top3')
        pd.DataFrame(age_avg_consumption, columns=['人均消费额(元)']).to_excel(writer, sheet_name='年龄人均消费')
        age_item_ratio.to_excel(writer, sheet_name='年龄消费偏好')
    
    print("✅ 分析结果已导出到：咖啡店数据分析结果.xlsx")

if __name__ == "__main__":
    main()
