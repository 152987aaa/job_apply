# TRAE SBTI - 程序员人格测试

## 项目概述

TRAE SBTI 是为 TRAE CN 社区设计的程序员人格测试，基于 GitHub 上的 SBTI 项目改编。

## 文件结构

```
TRAE_SBTI/
├── index.html    # 主页面（完整源码）
└── README.md     # 本说明文档
```

## 维度系统设计

### 五大模型（15维度）

| 模型 | 维度 | 说明 |
|------|------|------|
| **Code模型** | C1-C3 | 编程风格与习惯 |
| **AI模型** | A1-A3 | AI工具使用态度 |
| **学习模型** | L1-L3 | 学习方式与成长心态 |
| **社区模型** | S1-S3 | 社区互动模式 |
| **效率模型** | E1-E3 | 工作效率与时间管理 |

### 具体维度

```
Code模型
├── C1: 代码风格（追求完美 vs 快速实现）
├── C2: 技术广度（深耕 vs 广度）
└── C3: 问题解决（独立 vs 求助）

AI模型
├── A1: AI依赖度（辅助 vs 主导）
├── A2: AI信任度（怀疑 vs 信任）
└── A3: AI学习观（替代学习 vs 辅助学习）

学习模型
├── L1: 学习方式（理论 vs 实践）
├── L2: 成长心态（固定思维 vs 成长思维）
└── L3: 知识分享（保守 vs 开放）

社区模型
├── S1: 社区参与（潜水 vs 活跃）
├── S2: 互助态度（索取 vs 贡献）
└── S3: 反馈风格（批评 vs 建设性）

效率模型
├── E1: 工作节奏（计划驱动 vs 随性）
├── E2: 时间管理（精细 vs 粗放）
└── E3: 任务处理（专注 vs 多任务）
```

## 12种人格类型

| 代码 | 名称 | 简介 |
|------|------|------|
| WIZARD | 技术 wizard | 代码如魔法，一切皆可实现 |
| CRAFTSMAN | 代码匠人 | 慢工出细活，代码即作品 |
| HACKER | 黑客思维 | 规则是用来打破的 |
| COPILOT | AI协作者 | AI是我的第二大脑 |
| STUDENT | 终身学习者 | 学无止境，保持好奇 |
| PRAGMATIST | 实用主义者 | 够用就好，快速交付 |
| MENTOR | 技术导师 | 独乐乐不如众乐乐 |
| LURKER | 沉默观察者 | 默默观察，深度思考 |
| BUILDER | 全栈建造者 | 从零到一，创造一切 |
| ARCHITECT | 架构师思维 | 不谋全局者，不足谋一域 |
| DEBUGGER | 调试专家 | Bug虐我千百遍，我待Bug如初恋 |
| NINJA | 代码忍者 | 事了拂衣去，深藏身与名 |

## 自定义说明

### 需要修改的内容

如需进一步自定义，修改 `index.html` 中的 JavaScript 部分：

| 变量名 | 说明 |
|--------|------|
| `dimensionMeta` | 维度定义 |
| `dimensionOrder` | 维度显示顺序 |
| `questions` | 30道测试题目 |
| `TYPE_LIBRARY` | 12种人格的描述 |
| `NORMAL_TYPES` | 人格匹配模式 |
| `DIM_EXPLANATIONS` | 维度等级解释 |
| `TYPE_IMAGES` | 人格图片URL |

### 人格图片

当前使用 placeholder 图片，如需替换：

1. 准备12张程序员风格的图片
2. 将图片放入 `image/` 目录
3. 修改 `TYPE_IMAGES` 中的 URL 为本地路径

```javascript
const TYPE_IMAGES = {
  "WIZARD": "./image/wizard.png",
  // ...
};
```

## 使用方法

直接在浏览器中打开 `index.html` 即可使用。

## 部署

可部署到任意静态文件服务器：
- GitHub Pages
- Vercel
- Netlify
- Nginx
- Apache