# 本地复现说明

如果希望别人本地跑出的结果和当前机器一致，需要同时固定三件事：

1. 代码版本一致。
2. Python 与依赖版本一致。
3. 输入数据与关键中间结果一致。

推荐流程：

```powershell
& 'D:\anaconda\python.exe' -m pip install -r .\skills\factor-timing-advisor\requirements.lock.txt
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py update
```

日常追溯以 `workspace/runs/default/results/report/update_status.json` 和 `update_history.jsonl` 为准；它们记录本次输入、代码和关键输出指纹。旧的固定 `reproducibility_manifest.json` 只适合核对某个冻结版本，不应作为每日新增数据前的必过门槛。

如果你更新了数据，并且确认这是新的标准结果，重新生成指纹：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\check_reproducibility.py write --manifest references/reproducibility_manifest.json
```

注意：如果不提交 `workspace/data/` 和 `workspace/runs/`，则需要用其他方式分发同一份数据包。只同步 `.py` 文件不能保证结果一致。
