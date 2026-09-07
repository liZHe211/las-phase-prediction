# 公网部署说明

## 推荐方案

使用 GitHub + Streamlit Community Cloud。部署完成后会获得一个可以发给评委和其他人的 `https://*.streamlit.app` 网址，网页标题为“基于真实数据驱动的 LAS 微晶玻璃晶相智能预测与目标晶相设计系统”。

## 上传前检查

公开仓库应至少包含：

```text
app.py
requirements.txt
data/data_merged.csv
data/data_audit_report.md
.streamlit/config.toml
```

`.gitignore` 已排除本地的 `data/las_lab.db` 和 `data/uploads/`，不会把本机样品记录和上传图片一起发布。

## Streamlit Cloud 操作

1. 在 GitHub 新建一个公开仓库，例如 `las-phase-prediction`。
2. 将本项目文件上传到仓库根目录，并确认 `app.py`、`requirements.txt` 和 `data/data_merged.csv` 均已上传。
3. 打开 `https://share.streamlit.io/`，使用 GitHub 登录。
4. 选择 **Create app**，填写仓库、分支 `main` 和入口文件 `app.py`。
5. 在 App URL 中设置一个便于报名填写的英文短名称，例如 `las-phase-prediction`。
6. 等待构建完成，得到类似下面的公网地址：

```text
https://las-phase-prediction.streamlit.app
```

## 重要说明

- 这是公开演示系统，仓库中的 CSV 数据也会对外可见；确认数据可以公开后再上传。
- 免费云端实例重启后，本地 SQLite 样品记录可能被清空或恢复初始状态，因此当前样品记录适合演示，不适合长期实验数据归档。
- 预测结果是当前数据集上的科研筛选结果，不能替代 XRD 鉴定和实际热处理实验。
- 若不希望公开原始数据，应改成私有仓库，并确认云服务账户和部署方案支持私有仓库访问。
