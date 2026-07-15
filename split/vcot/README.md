# VCoT split metadata

These files are official VCoT split metadata for the Grasp-Anything dataset.
The two small evaluation splits are included:

- `test_seen.csv`
- `test_unseen.csv`

The approximately 27 MB `train.csv` is intentionally excluded. Download it
from the official VCoT-Grasp repository before training:

```bash
curl -L \
  https://raw.githubusercontent.com/zhanghr2001/VCoT-Grasp/main/split/vcot/train.csv \
  -o split/vcot/train.csv
```

Alternatively, clone the official repository and copy
`VCoT-Grasp/split/vcot/train.csv` into this directory.

Each row stores `grasp_id, object_name, scene_description`. The images and
annotations are not copied into this repository; set `DATA.root_path` to the
local Grasp-Anything root containing `image/`, `positive_grasp/`, and `mask/`.

Source: [official VCoT-Grasp repository](https://github.com/zhanghr2001/VCoT-Grasp/tree/main/split/vcot)
(MIT License).
