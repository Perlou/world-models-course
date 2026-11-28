"""
Phase 11 · 第 09 课：其他前沿（自动驾驶 / 导航 / 交互式视频）与论文归位练习
=========================================================================

学习目标：
    1. 将自动驾驶（GAIA-1 / GAIA-2 / Vista）、导航（NWM）、交互式视频（Cosmos / UniSim /
       Genie 2&3 / Dreamer 4）与机器人世界模型放进同一张五维地图
    2. 实现 classify_paper() 归类工具，按 CONCEPT.md 第 9.5 节五个问题自动给出用法定位与必问证据
    3. 完成 4 篇典型摘要的归位练习并核对参考答案

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 9 节

运行方式：
    uv run python src/phase-11-robot-world-models/09-other-frontiers.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/09-other-frontiers.py

预计运行时间：< 2 秒
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PAPERS = [
    ("GAIA-1 (2023)", "视频+文本+驾驶动作", "离散 Token (自回归)", "仿真器 / 数据引擎", "自动驾驶"),
    ("GAIA-2 (2025)", "多相机+驾驶控制", "连续隐空间扩散", "仿真器 / 数据引擎", "自动驾驶"),
    ("Vista (2024)", "驾驶指令/轨迹", "视频扩散", "仿真器 / 规划打分", "自动驾驶"),
    ("Navigation WM (2024)", "导航动作", "条件 DiT 视频扩散", "决策时规划 (目标图像)", "移动导航"),
    ("UniPi (2023)", "语言指令", "像素视频扩散 (+IDM)", "当策略 (两阶段)", "机器人操作"),
    ("UWM / UVA (2025)", "历史观测+指令", "视频+动作联合扩散", "当策略 / 世界模型多用", "机器人操作"),
    ("WorldGym / WorldEval", "策略动作", "动作条件视频扩散", "当仿真器 (策略评估)", "机器人操作"),
    ("DreamGen (2025)", "起始帧+语言指令", "视频扩散 (+IDM 伪标)", "当数据引擎", "机器人操作"),
    ("DayDreamer (2022)", "机器人动作", "RSSM 隐状态+像素重建", "真机在线 RL 学习", "四足/机械臂"),
    ("V-JEPA 2-AC (2025)", "末端位姿动作", "冻结视频表征 (JEPA)", "真机零样本 CEM 规划", "机器人操作"),
    ("Dreamer 4 (2025)", "混合无标+键鼠", "隐空间 Shortcut Forcing", "离线想象 RL 训练", "通用交互/游戏"),
]


def classify_paper(condition, target, consumer, evidence):
    """CONCEPT.md 第 9.5 节：按四个核心维度将任意新论文归入世界模型地图，并列出应追问的批判性问题。"""
    if consumer == "policy":
        role = "用法一：当策略"
        checks = ["推理延迟能否跟上控制频率？", "逆动力学/动作头在接触细节上是否准确？"]
    elif consumer == "evaluator":
        role = "用法二：当仿真器（策略评估）"
        checks = ["在几个真实策略上验证过 Pearson / Spearman / MMRV？", "是否含失败策略数据以防幻觉成功？"]
    elif consumer == "data_engine":
        role = "用法三：当数据引擎"
        checks = ["真实数据增多后生成数据的增益是否依然为正？", "最终是否回到真实环境评估而非自己评自己？"]
    elif consumer == "planner":
        role = "用法四：决策时规划（MPC/CEM）"
        checks = ["单步规划耗时支持几 Hz 控制？", "表征距离是否存在局部极小或漏细节？"]
    else:
        role = "用法四：想象中在线/离线强化学习"
        checks = ["奖励如何自动获取？", "真机探索的复位与安全约束如何保障？"]

    if target == "pixel_or_video":
        paradigm = "生成式世界模型（Phase 8/9）"
    elif target == "representation":
        paradigm = "JEPA 表征预测世界模型（Phase 10）"
    else:
        paradigm = "World-Action Model 联合建模（Phase 11 第 5 节）"

    risk = "证据仅停留在视觉指标（FVD/PSNR），尚不足以证明对控制决策有效！" if evidence == "visual_only" else "已包含下游控制/排序证据，需重点核查回合数与 95% 置信区间。"
    return {"role": role, "paradigm": paradigm, "checks": checks, "evidence_note": risk, "condition": condition}


def landscape_table():
    print("=" * 88)
    print("第一部分：跨领域世界模型全景地图（操作 / 自动驾驶 / 导航 / 通用交互）")
    print("=" * 88)
    print(f"  {'工作':22s} | {'条件输入':18s} | {'预测对象':22s} | {'核心用法':20s} | {'领域'}")
    print("  " + "-" * 84)
    for p in PAPERS:
        print(f"  {p[0]:22s} | {p[1]:18s} | {p[2]:22s} | {p[3]:20s} | {p[4]}")


def reading_exercise():
    print("\n" + "=" * 88)
    print("第二部分：新论文归位阅读练习（基于 CONCEPT.md 第 9.5 节五问法）")
    print("=" * 88)
    cases = [
        (
            "案例 A：某自动驾驶论文用多视角隐空间扩散以方向盘/油门为条件生成 30 秒罕见加塞视频，只汇报了 FVD=85。",
            dict(condition="robot_action", target="pixel_or_video", consumer="evaluator", evidence="visual_only"),
        ),
        (
            "案例 B：某机械臂论文冻结 100 万小时视频预训练编码器，用 50 小时动作数据训练表征预测器，接 CEM 规划目标图。",
            dict(condition="robot_action", target="representation", consumer="planner", evidence="real_robot"),
        ),
        (
            "案例 C：某论文在一个 Transformer 中对动作块和未来观测分别加独立噪声，可混用无动作视频并跳过视频推理。",
            dict(condition="obs_and_lang", target="joint_video_action", consumer="policy", evidence="sim_and_real"),
        ),
    ]
    for desc, kw in cases:
        res = classify_paper(**kw)
        print(f"\n  {desc}")
        print(f"    → 归类：{res['role']} ｜ 路线：{res['paradigm']}")
        print(f"    → 证据评价：{res['evidence_note']}")
        print(f"    → 必追问：{'；'.join(res['checks'])}")


def main():
    landscape_table()
    reading_exercise()
    print("\n本章全部 9 节讲义完成！接下来：实战项目 projects/phase-11-wm-policy-evaluation/")


if __name__ == "__main__":
    main()
