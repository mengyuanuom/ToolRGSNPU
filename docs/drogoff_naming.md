# DROG-OFF naming

`DROG-OFF` is the architecture name. Two independent things had previously
been called “V2”: the Grasp-Tools dataset release and the offset target/decoder
protocol. They are now named separately.

| Config | Dataset | Offset protocol | Notes |
| --- | --- | --- | --- |
| `config/grasp_tools/drogoff_grasp_tools_v2.yaml` | Grasp-Tools Dataset V2 | Offset V1 | Standard eight-NPU Grasp-Tools profile |
| `config/grasp_tools/drogoff_offset_v2.yaml` | Grasp-Tools Dataset V2 | Dense Offset V2 | Lightweight region-regression profile |
| `config/ocid_vlg/drogoff.yaml` | OCID-VLG | Offset V1 | Standard OCID-VLG profile |
| `config/ocid_vlg/drogoff_offset_v2.yaml` | OCID-VLG | Dense Offset V2 | Lightweight region-regression profile |
| `config/realvlg/drogoff.yaml` | RealVLG | Offset V1 | RealVLG 10% profile |
| `config/realvlg/drogoff_offset_v2.yaml` | RealVLG | Dense Offset V2 | RealVLG 10% with official evaluator |
| `config/vcot/drogoff.yaml` | VCoT | Offset V1 | Checkpoint-compatible protocol |
| `config/vcot/drogoff_offset_v2.yaml` | VCoT | Dense Offset V2 | Full quality-region target and scale-relative decode |

The authoritative field is `DATA.offset_version`. Old configs and checkpoints
without this metadata are interpreted as `v1`. The architecture remains
`MODEL.architecture: drogoff` for both protocols.

## 中文说明

- `grasp_tools_v2` 只表示 **Grasp-Tools 数据集第二版**。
- `offset_v1` 是旧的固定半径中心偏移监督，兼容原有权重。
- `offset_v2` 才表示新的稠密偏移监督与对应解码协议。
- 四个数据集目录都提供独立的 `drogoff_offset_v2.yaml`，旧 V1 配置不变。
- 判断版本请看 `DATA.offset_version`，不要只看 `DROG-OFF V2` 这种简称。
