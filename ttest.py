import scipy.stats as stats  # 导入scipy的统计模块
import pandas as pd  # 导入pandas模块用于数据处理
import os
import numpy as np  # 导入numpy用于计算标准差等
from argparse import ArgumentParser  # 导入命令行参数解析模块

def calculate_cohens_d(group1, group2):
    """
    计算Cohen's d效应量
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate the pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    if pooled_se == 0:
        return 0
    d = (np.mean(group1) - np.mean(group2)) / pooled_se
    return d

# 确保这段代码作为主程序运行时才执行
if __name__ == "__main__":
    parser = ArgumentParser()  # 创建一个参数解析器
    # 添加一个输入文件参数，默认值为指定的CSV文件路径
    # parser.add_argument("-if", "--input_file", type=str, default="./evaluated_letters/zhipuai/cbg/all_2_para_w_zhipuai-eval_hallucination-eval.csv")
    # parser.add_argument("-if", "--input_file", type=str, default="./evaluated_letters/deepseek/cbg/all_2_para_w_deepseek-eval.csv")
    parser.add_argument("-if", "--input_file", type=str, default="./evaluated_letters/clg_letters-eval.csv")
    # 添加一个布尔型参数，用于判断是否评估幻觉部分
    parser.add_argument('--eval_hallucination_part', action='store_true')
    parser.add_argument("-of", "--output_folder", type=str, default=r"c:\工作\论文\大模型性别偏见\数据分析与知识发现\code1-19\result\Language Style")
    args = parser.parse_args()  # 解析命令行参数

    df = pd.read_csv(args.input_file)  # 读取CSV文件到DataFrame

    # 根据性别字段对数据进行分组，支持'male', 'female'或'm', 'f'
    # if df['gender'][0] in ['male', 'female']:
    #     df_m = df[df['gender'] == 'male']  # 男性数据
    #     df_f = df[df['gender'] == 'female']  # 女性数据
    # else:
    #     df_m = df[df['gender'] == 'm']  # 男性数据
    #     df_f = df[df['gender'] == 'f']  # 女性数据
    
    first_gender = df['gender'].iloc[0]
    if first_gender in ['男性', '女性']:
        df_m = df[df['gender'] == '男性']  # 男性数据
        df_f = df[df['gender'] == '女性']  # 女性数据
    elif first_gender in ['男', '女']:
        df_m = df[df['gender'] == '男']  # 男性数据
        df_f = df[df['gender'] == '女']  # 女性数据
    elif first_gender in ['male', 'female', 'm', 'f']:
        df_m = df[df['gender'].isin(['male', 'm'])]
        df_f = df[df['gender'].isin(['female', 'f'])]
    else:
        raise ValueError(f"Unknown gender format: {first_gender}")

    # 遍历指定的推论类型
    # 更新分析指标：移除旧的二分类指标，添加新的平均情感强度指标
    # 注意：如果classifier.py中没有计算formality，则per_for/con_for可能不存在，这里最好做个检查
    columns_to_analyze = ["avg_sentiment_intensity", "per_for", "con_for"]

    # Auto-detect hallucination columns if not explicitly set
    if not args.eval_hallucination_part:
        has_hallucination_cols = False
        for col in columns_to_analyze:
            if f"{col}_1" in df.columns:
                has_hallucination_cols = True
                break
        
        if has_hallucination_cols:
            print("检测到原始/幻觉对比列（_1后缀），自动开启幻觉评估模式 (Enabling hallucination evaluation mode).")
            args.eval_hallucination_part = True
    
    # 指标中文名映射
    metric_names = {
        "avg_sentiment_intensity": "平均情感强度 (Avg Sentiment Intensity)",
        "per_for": "正式性占比 (Percentage of Formality)",
        "con_for": "正式性置信度 (Confidence of Formality)"
    }
    
    # 设置输出目录
    output_dir = args.output_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径
    base_name = os.path.basename(args.input_file).replace('.csv', '-analysis_report.md')
    output_file = os.path.join(output_dir, base_name)
    
    # 用于存储统计结果以保存为CSV
    csv_results = []

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 分析结果报告\n\n")
        f.write(f"**数据源文件:** `{args.input_file}`\n\n")
        f.write(f"**分析日期:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 样本概况\n")
        f.write(f"- **总样本数**: {len(df)}\n")
        f.write(f"- **男性样本数**: {len(df_m)}\n")
        f.write(f"- **女性样本数**: {len(df_f)}\n\n")
        
        f.write("---\n\n")

        for inference in columns_to_analyze:
            if inference not in df.columns:
                msg = f"> **跳过指标 `{inference}`**: 在数据表中未找到该列。\n"
                print(msg.replace("**", "").replace("`", "").strip())
                f.write(msg + "\n")
                continue
                
            metric_name_cn = metric_names.get(inference, inference)
            f.write(f"## 指标分析: `{inference}` ({metric_name_cn})\n\n")

            # Always perform Male vs Female comparison
            if True:
                # 提取男性和女性的推论数据
                per_f = df_f[inference].tolist()    
                per_m = df_m[inference].tolist()

                # 进行独立样本t检验，比较男性和女性的推论数据
                res = stats.ttest_ind(a=per_m, b=per_f, equal_var=True, alternative='two-sided')
                statistic, pvalue = res[0], res[1]  # 获取统计量和p值
                
                # 计算Cohen's d
                d = calculate_cohens_d(per_m, per_f)

                # 打印推论类型、统计量和p值
                print("指标: {}\n统计量: {}\np值: {}\nCohen's d: {}".format(inference, statistic, pvalue, d))
                
                # 写入Markdown
                f.write(f"- **对比组**: 男性 vs 女性\n")
                f.write(f"- **检验类型**: 独立样本 t 检验 (双尾/two-sided)\n")
                f.write(f"- **统计量 (Statistic)**: `{statistic:.4f}`\n")
                f.write(f"- **P值 (P-value)**: `{pvalue:.4e}`\n")
                f.write(f"- **Cohen's d**: `{d:.4f}`\n")
                
                # 添加到CSV结果列表
                csv_results.append({
                    "metric": inference,
                    "comparison": "Male vs Female",
                    "statistic": statistic,
                    "p_value": pvalue,
                    "cohens_d": d,
                    "significant": pvalue < 0.05
                })

                if pvalue < 0.05:
                    direction = "男性 > 女性" if statistic > 0 else "男性 < 女性"
                    f.write(f"- **结论**: **差异显著**。统计结果拒绝原假设，表明在 `{inference}` 指标上，男性和女性样本存在显著差异 ({direction}) (p < 0.05)。\n")
                else:
                    f.write(f"- **结论**: **差异不显著**。统计结果无法拒绝原假设，没有足够证据表明在 `{inference}` 指标上男性和女性样本存在显著差异 (p >= 0.05)。\n")
                f.write("\n")

            if args.eval_hallucination_part:
                # 如果评估幻觉部分，提取幻觉和原始内容的推论数据
                hal_f = df_f[inference].tolist()   
                ori_f = df_f['{}_1'.format(inference)].tolist()
                hal_m = df_m[inference].tolist()
                ori_m = df_m['{}_1'.format(inference)].tolist()

                # 1. 男性幻觉 vs 原始内容
                res1 = stats.ttest_ind(a=hal_m, b=ori_m, equal_var=True, alternative='two-sided')
                statistic1, pvalue1 = res1[0], res1[1]
                d1 = calculate_cohens_d(hal_m, ori_m)
                
                print("男性生成信件幻觉内容与原始内容的推论。推论类型: {}\n统计量: {}\np值: {}\nCohen's d: {}".format(inference, statistic1, pvalue1, d1))
                
                csv_results.append({
                    "metric": inference,
                    "comparison": "Male: Hallucination vs Original",
                    "statistic": statistic1,
                    "p_value": pvalue1,
                    "cohens_d": d1,
                    "significant": pvalue1 < 0.05
                })

                f.write(f"### 男性样本: 幻觉内容 vs 原始内容\n")
                f.write(f"- **统计量**: `{statistic1:.4f}`\n")
                f.write(f"- **P值**: `{pvalue1:.4e}`\n")
                f.write(f"- **Cohen's d**: `{d1:.4f}`\n")
                if pvalue1 < 0.05:
                    direction = "幻觉 > 原始" if statistic1 > 0 else "幻觉 < 原始"
                    f.write(f"- **结论**: **差异显著** ({direction})\n")
                else:
                    f.write(f"- **结论**: 差异不显著\n")
                f.write("\n")

                # 2. 女性幻觉 vs 原始内容
                # 统一检验逻辑：检验幻觉内容是否显著不同于原始内容
                res2 = stats.ttest_ind(a=hal_f, b=ori_f, equal_var=True, alternative='two-sided')
                statistic2, pvalue2 = res2[0], res2[1]
                d2 = calculate_cohens_d(hal_f, ori_f)
                
                print("女性生成信件幻觉内容与原始内容的推论。推论类型: {}\n统计量: {}\np值: {}\nCohen's d: {}".format(inference, statistic2, pvalue2, d2))

                csv_results.append({
                    "metric": inference,
                    "comparison": "Female: Hallucination vs Original",
                    "statistic": statistic2,
                    "p_value": pvalue2,
                    "cohens_d": d2,
                    "significant": pvalue2 < 0.05
                })

                f.write(f"### 女性样本: 幻觉内容 vs 原始内容\n")
                f.write(f"- **统计量**: `{statistic2:.4f}`\n")
                f.write(f"- **P值**: `{pvalue2:.4e}`\n")
                f.write(f"- **Cohen's d**: `{d2:.4f}`\n")
                if pvalue2 < 0.05:
                    direction = "幻觉 > 原始" if statistic2 > 0 else "幻觉 < 原始"
                    f.write(f"- **结论**: **差异显著** ({direction})\n")
                else:
                    f.write(f"- **结论**: 差异不显著\n")
                f.write("\n")

                # 3. 男性幻觉 vs 女性幻觉 (新增)
                res3 = stats.ttest_ind(a=hal_m, b=hal_f, equal_var=True, alternative='two-sided')
                statistic3, pvalue3 = res3[0], res3[1]
                d3 = calculate_cohens_d(hal_m, hal_f)
                
                print("幻觉内容性别对比（男 vs 女）。推论类型: {}\n统计量: {}\np值: {}\nCohen's d: {}".format(inference, statistic3, pvalue3, d3))

                csv_results.append({
                    "metric": inference,
                    "comparison": "Hallucination: Male vs Female",
                    "statistic": statistic3,
                    "p_value": pvalue3,
                    "cohens_d": d3,
                    "significant": pvalue3 < 0.05
                })

                f.write(f"### 幻觉内容性别对比: 男性 vs 女性\n")
                f.write(f"- **检验类型**: 独立样本 t 检验 (双尾/two-sided)\n")
                f.write(f"- **统计量**: `{statistic3:.4f}`\n")
                f.write(f"- **P值**: `{pvalue3:.4e}`\n")
                f.write(f"- **Cohen's d**: `{d3:.4f}`\n")
                if pvalue3 < 0.05:
                    direction = "男性幻觉 > 女性幻觉" if statistic3 > 0 else "男性幻觉 < 女性幻觉"
                    f.write(f"- **结论**: **差异显著** ({direction})\n")
                else:
                    f.write(f"- **结论**: 差异不显著 (男性幻觉 <= 女性幻觉)\n")
                f.write("\n")

                # 4. 偏见放大分析 (Bias Amplification Analysis)
                # 计算原始内容的性别偏见 (Original Content Male vs Female)
                # 注意：ori_m 和 ori_f 已经在前面提取过了
                res_orig = stats.ttest_ind(a=ori_m, b=ori_f, equal_var=True, alternative='two-sided')
                statistic_orig, pvalue_orig = res_orig[0], res_orig[1]
                d_orig = calculate_cohens_d(ori_m, ori_f)
                
                # 幻觉内容的性别偏见 (d3)
                d_hall = d3
                
                # 效应量差值
                d_diff = d_hall - d_orig
                
                print("偏见放大分析。推论类型: {}\n原文偏见(d_orig): {}\n幻觉偏见(d_hall): {}\n差值(d_diff): {}".format(inference, d_orig, d_hall, d_diff))

                csv_results.append({
                    "metric": inference,
                    "comparison": "Bias Amplification (d_hall - d_orig)",
                    "statistic": d_diff, # 这里用差值代替统计量
                    "p_value": np.nan, # 没有直接的p值
                    "cohens_d": d_diff,
                    "significant": abs(d_diff) > 0.1 # 简单的阈值判定，仅供参考
                })

                f.write(f"### 偏见放大分析: 幻觉偏见 vs 原文偏见\n")
                f.write(f"- **原文性别偏见 (Cohen's d_orig)**: `{d_orig:.4f}` (Male vs Female)\n")
                f.write(f"- **幻觉性别偏见 (Cohen's d_hall)**: `{d_hall:.4f}` (Male vs Female)\n")
                f.write(f"- **效应量差值 (Δd = d_hall - d_orig)**: `{d_diff:.4f}`\n")
                
                if d_diff > 0:
                    f.write(f"- **结论**: 幻觉内容中的性别偏见相比原文 **增强** 了 (Δd > 0)。\n")
                elif d_diff < 0:
                    f.write(f"- **结论**: 幻觉内容中的性别偏见相比原文 **减弱** 或 **反转** 了 (Δd < 0)。\n")
                else:
                    f.write(f"- **结论**: 幻觉内容中的性别偏见与原文持平。\n")
                f.write("\n")
        
    print(f"\n分析结果已保存至: {output_file}")
    
    # Save CSV results
    if csv_results:
        results_df = pd.DataFrame(csv_results)
        csv_output_path = os.path.join(output_dir, base_name.replace('-analysis_report.md', '-ttest_results.csv'))
        results_df.to_csv(csv_output_path, index=False)
        print(f"统计结果CSV已保存至: {csv_output_path}")
